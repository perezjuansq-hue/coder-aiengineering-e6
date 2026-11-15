"""Ingesta: lee los documentos de ./data, los fragmenta y los sube a un índice serverless de Pinecone.

Uso:
    python ingest.py            # indexa solo si los documentos o la configuración cambiaron
    python ingest.py --force    # vacía el namespace y vuelve a indexar todo
    python ingest.py --preview  # muestra cómo quedan los chunks sin generar embeddings
"""
import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Index, NotFoundException, Pinecone

from setup_index import DIMENSION, NAMESPACE, asegurar_indice, cliente_pinecone

logger = logging.getLogger("ingest")

DIR_DATA = Path("data")

# El MISMO modelo se usa para indexar y para consultar: se guarda como tag del índice y se verifica
MODELO_EMBEDDINGS = "text-embedding-3-small"
# text-embedding-3-small usa el tokenizer cl100k_base: medimos los chunks con ese mismo tokenizer
ENCODING = "cl100k_base"
# La consigna sugiere ~500-800 tokens. Con 500, la mediana real quedaba en ~415 (los puntos cortos
# de la norma no llegan al máximo), así que subimos a 700 para que la mayoría caiga en el rango.
CHUNK_SIZE = 700     # tokens
CHUNK_OVERLAP = 100  # tokens
LOTE_UPSERT = 100    # vectores por request a Pinecone
# Versión del esquema de metadatos: si se agrega o cambia un campo, se sube y la huella fuerza reindexar
ESQUEMA = 2

# Etiqueta de categoría por documento (código del texto ordenado del BCRA): permite filtrar búsquedas
CATEGORIAS = {
    "t-pusf": "proteccion_usuarios",
    "t-depaho": "cuentas_y_depositos",
    "t-tasint": "tasas_y_credito",
    "t-efemin": "regulacion_monetaria",
}

PATRON_MARCADOR = re.compile(r"<!--\s*((?:pagina|documento):.*?)\s*-->")  # marcadores de prepare_data.py
PATRON_TITULO = re.compile(r"^(#{2,4})\s+(.*)$")


# --- Lectura de los documentos -----------------------------------------------------------------

def leer_marcador(linea: str) -> dict | None:
    """'<!-- pagina: 7 | vigencia: 23/02/2024 -->' -> {'pagina': '7', 'vigencia': '23/02/2024'}"""
    m = PATRON_MARCADOR.fullmatch(linea.strip())
    if not m:
        return None
    pares = (parte.split(":", 1) for parte in m.group(1).split(" | "))
    return {clave.strip(): valor.strip() for clave, valor in pares}


def dividir_en_bloques(texto: str) -> tuple[dict, list[dict]]:
    """Divide un documento en bloques por sección (##) y punto (###).

    Cada bloque conserva los títulos que lo contienen (para los metadatos) y los marcadores de
    página que tiene adentro (para saber en qué página del PDF está cada fragmento).
    """
    meta_doc: dict = {}
    bloques: list[dict] = []
    titulos = {"seccion": "", "punto": ""}
    actual = {"seccion": "", "punto": "", "lineas": []}

    def cerrar_bloque():
        if any(linea.strip() and not PATRON_MARCADOR.fullmatch(linea.strip()) for linea in actual["lineas"]):
            bloques.append(dict(actual))

    for linea in texto.splitlines():
        marcador = leer_marcador(linea)
        if marcador and "documento" in marcador:
            meta_doc = marcador
            continue
        if linea.startswith("# "):  # título del documento: ya está en los metadatos
            continue

        titulo = PATRON_TITULO.match(linea)
        if titulo and len(titulo.group(1)) in (2, 3):
            cerrar_bloque()
            if len(titulo.group(1)) == 2:
                titulos = {"seccion": titulo.group(2), "punto": ""}
            else:
                titulos["punto"] = titulo.group(2)
            # Si el bloque anterior terminaba con un marcador de página, lo arrastramos al nuevo
            arrastre = [l for l in actual["lineas"][-3:] if PATRON_MARCADOR.fullmatch(l.strip())]
            actual = {**titulos, "lineas": arrastre[-1:] + [linea]}
        else:
            actual["lineas"].append(linea)
    cerrar_bloque()
    return meta_doc, bloques


def agrupar_bloques(bloques: list[dict], contar_tokens) -> list[dict]:
    """Une puntos cortos consecutivos de la misma sección hasta llegar a CHUNK_SIZE tokens.

    El splitter solo corta lo que es grande; sin este paso, cada punto breve de la norma sería un
    chunk de pocas decenas de tokens, con poco contexto semántico. Nunca se mezclan secciones.
    """
    grupos: list[dict] = []
    for bloque in bloques:
        tokens = contar_tokens("\n".join(bloque["lineas"]))
        previo = grupos[-1] if grupos else None
        if previo and previo["seccion"] == bloque["seccion"] and previo["tokens"] + tokens <= CHUNK_SIZE:
            previo["lineas"] += bloque["lineas"]
            previo["tokens"] += tokens
            if bloque["punto"] and bloque["punto"] not in previo["puntos"]:
                previo["puntos"].append(bloque["punto"])
        else:
            grupos.append({"seccion": bloque["seccion"], "puntos": [bloque["punto"]] if bloque["punto"] else [],
                           "lineas": list(bloque["lineas"]), "tokens": tokens})
    return grupos


def crear_chunks(archivo: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    meta_doc, bloques = dividir_en_bloques(archivo.read_text(encoding="utf-8"))
    bloques = agrupar_bloques(bloques, splitter._length_function)
    documento = meta_doc.get("documento", archivo.stem)
    url = meta_doc.get("fuente", "")
    doc_id = Path(url).stem or archivo.stem  # "t-depaho": identificador estable del documento
    pagina = {"pagina": "s/d", "vigencia": "s/d", "comunicacion": "s/d"}
    chunks = []

    for bloque in bloques:
        for fragmento in splitter.split_text("\n".join(bloque["lineas"])):
            # Página donde empieza el fragmento: la última vista antes de él (o la primera que contiene
            # si arranca con un marcador). Página final: el último marcador dentro del fragmento.
            marcadores = [leer_marcador(m.group(0)) for m in PATRON_MARCADOR.finditer(fragmento)]
            if fragmento.lstrip().startswith("<!--") and marcadores:
                pagina = marcadores[0]
            pagina_inicio = pagina
            if marcadores:
                pagina = marcadores[-1]

            texto = PATRON_MARCADOR.sub("", fragmento)
            texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
            if len(texto) < 40:  # restos sin contenido (por ejemplo, un título suelto)
                continue

            # "Contextual chunk header": el encabezado viaja con el texto, así el embedding y el LLM
            # saben a qué norma y punto pertenece el fragmento aunque el texto no lo diga.
            punto = " / ".join(bloque["puntos"])
            encabezado = f"[{documento} | {bloque['seccion']} | {punto}]".replace(" | ]", "]")
            chunks.append(Document(
                page_content=f"{encabezado}\n{texto}",
                metadata={
                    "doc_id": doc_id,
                    "categoria": CATEGORIAS.get(doc_id, "otros"),
                    "fuente": archivo.name,
                    "documento": documento,
                    "seccion": bloque["seccion"],
                    "punto": punto,                     # para mostrar: "1.8. ... / 1.9. ..."
                    "puntos": bloque["puntos"],         # lista: para citar y filtrar por punto
                    "pagina_inicio": int(pagina_inicio["pagina"]) if pagina_inicio["pagina"].isdigit() else -1,
                    "pagina_fin": int(pagina["pagina"]) if pagina["pagina"].isdigit() else -1,
                    "vigencia": pagina_inicio.get("vigencia", "s/d"),
                    "comunicacion": pagina_inicio.get("comunicacion", "s/d"),
                    "texto_ordenado_al": meta_doc.get("texto_ordenado_al", "s/d"),
                    "url": url,
                },
            ))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
        chunk.id = f"{doc_id}-{i:04d}"  # id determinístico: re-subir el mismo chunk lo pisa, no lo duplica
        chunk.metadata["chunk_id"] = chunk.id  # también en la metadata: para filtrar y deduplicar por id
    return chunks


def cargar_chunks() -> list[Document]:
    archivos = sorted([*DIR_DATA.glob("*.md"), *DIR_DATA.glob("*.txt")])
    if not archivos:
        raise FileNotFoundError(f"No hay documentos en {DIR_DATA}/. Corré primero: python prepare_data.py")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for archivo in archivos:
        chunks_archivo = crear_chunks(archivo, splitter)
        logger.info("%s: %d chunks", archivo.name, len(chunks_archivo))
        chunks += chunks_archivo
    return chunks


# --- Pinecone ----------------------------------------------------------------------------------

def huella_de_datos() -> str:
    """Hash del contenido de ./data y de la configuración de chunking/embeddings.

    Si cambia un documento o un parámetro, cambia la huella y hay que volver a indexar.
    """
    h = hashlib.sha256()
    h.update(json.dumps([ESQUEMA, MODELO_EMBEDDINGS, DIMENSION, ENCODING, CHUNK_SIZE, CHUNK_OVERLAP,
                         CATEGORIAS]).encode())
    for archivo in sorted([*DIR_DATA.glob("*.md"), *DIR_DATA.glob("*.txt")]):
        h.update(archivo.name.encode())
        h.update(archivo.read_bytes())
    return h.hexdigest()[:16]


def crear_embeddings() -> OpenAIEmbeddings:
    load_dotenv()
    return OpenAIEmbeddings(model=MODELO_EMBEDDINGS, dimensions=DIMENSION)


def contar_vectores(indice: Index) -> int:
    stats = indice.describe_index_stats()
    ns = stats.namespaces.get(NAMESPACE)
    return ns.vector_count if ns else 0


def a_vector(chunk: Document, valores: list[float]) -> dict:
    """Esquema de cada registro: el texto original viaja en la metadata ("texto"), así una sola
    consulta a Pinecone devuelve todo lo necesario, sin ir a buscar el contenido a otra base."""
    return {"id": chunk.id, "values": valores, "metadata": {**chunk.metadata, "texto": chunk.page_content}}


def abrir_indice() -> Index:
    """Abre el índice verificando que se haya indexado con el mismo modelo de embeddings."""
    pc = cliente_pinecone()
    nombre = asegurar_indice(pc)
    tags = pc.describe_index(nombre).tags or {}
    modelo_indexado = tags.get("embedding_model")
    if modelo_indexado != MODELO_EMBEDDINGS:
        raise RuntimeError(
            f"El índice se creó con '{modelo_indexado}' pero se está consultando con "
            f"'{MODELO_EMBEDDINGS}'. Volvé a indexar con: python ingest.py --force"
        )
    return pc.Index(nombre)


def indexar(forzar: bool = False) -> Index:
    """Sube los chunks si el namespace está vacío o los documentos cambiaron; si no, no hace nada."""
    pc = cliente_pinecone()
    nombre = asegurar_indice(pc)
    indice = pc.Index(nombre)

    # Pinecone no tiene "metadata de colección" como Chroma: guardamos la huella como tag del índice
    tags = dict(pc.describe_index(nombre).tags or {})
    existentes = contar_vectores(indice)
    if not forzar and tags.get("huella") == huella_de_datos() and existentes > 0:
        logger.info("Namespace '%s' actualizado (%d vectores, huella %s) -> no se reindexa",
                    NAMESPACE, existentes, tags["huella"])
        return abrir_indice()

    if existentes:
        motivo = "--force" if forzar else "los documentos o la configuración cambiaron"
        logger.info("Reindexando (%s): se vacía el namespace '%s' (%d vectores)", motivo, NAMESPACE, existentes)
        try:
            indice.delete(delete_all=True, namespace=NAMESPACE)
        except NotFoundException:
            pass

    chunks = cargar_chunks()
    logger.info("Generando embeddings de %d chunks con %s...", len(chunks), MODELO_EMBEDDINGS)
    valores = crear_embeddings().embed_documents([c.page_content for c in chunks])

    vectores = [a_vector(c, v) for c, v in zip(chunks, valores)]
    indice.upsert(vectors=vectores, namespace=NAMESPACE, batch_size=LOTE_UPSERT, show_progress=False)

    # Pinecone es "eventualmente consistente": el conteo puede tardar unos segundos en reflejar el upsert
    for _ in range(30):
        if contar_vectores(indice) == len(vectores):
            break
        time.sleep(1)
    tags.update({"embedding_model": MODELO_EMBEDDINGS, "huella": huella_de_datos(),
                 "chunk_size": str(CHUNK_SIZE), "chunk_overlap": str(CHUNK_OVERLAP)})
    pc.configure_index(nombre, tags=tags)
    logger.info("Namespace '%s' cargado: %d vectores en el índice '%s'", NAMESPACE, contar_vectores(indice), nombre)
    return indice


def mostrar_preview() -> None:
    import tiktoken

    enc = tiktoken.get_encoding(ENCODING)
    chunks = cargar_chunks()
    tamanos = sorted(len(enc.encode(c.page_content)) for c in chunks)
    en_rango = sum(500 <= t <= 800 for t in tamanos)
    print(f"\n{len(chunks)} chunks | tokens: mín {tamanos[0]}, mediana {tamanos[len(tamanos) // 2]}, "
          f"máx {tamanos[-1]} | en el rango 500-800: {en_rango} ({en_rango / len(tamanos):.0%})")
    ejemplo = next(c for c in chunks if "1,8 milímetros" in c.page_content)
    print(f"\n--- Ejemplo de chunk (id {ejemplo.id}) ---")
    print(ejemplo.page_content[:700], "...")
    print("\nMetadatos:", json.dumps(ejemplo.metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="reindexar aunque nada haya cambiado")
    parser.add_argument("--preview", action="store_true", help="mostrar los chunks sin indexar")
    args = parser.parse_args()

    if args.preview:
        mostrar_preview()
    else:
        indexar(forzar=args.force)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for nombre in ("httpx", "httpx2", "openai", "pinecone"):
        logging.getLogger(nombre).setLevel(logging.WARNING)
    main()
