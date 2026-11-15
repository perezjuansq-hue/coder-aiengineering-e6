"""Herramientas del agente: búsqueda técnica sobre la normativa del BCRA (RAG híbrido de la PE4) y
lectura exacta de un punto de la norma (simula una consulta a base de datos por clave).

El LLM elige qué herramienta usar SOLO a partir del nombre, el docstring y el esquema de argumentos:
por eso los docstrings explican cuándo usar cada una, qué devuelve y qué hacer si falla.

Las herramientas nunca lanzan excepciones hacia el agente: ante un error o un resultado pobre devuelven
un JSON con "estado" y una "sugerencia". Eso habilita el ciclo de retorno (reintentar o pedir aclaración).

Uso (prueba aislada de las herramientas, sin LLM):
    python -m tools.normativa
"""
import asyncio
import json
import logging
import re
import sys
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.tools import tool

from rag_system import RAGSystem

logger = logging.getLogger("tools")

# Mismo umbral que la PE4: por debajo de este coseno, el mejor fragmento no es del tema consultado
UMBRAL_RELEVANCIA = 0.40
MAX_CARACTERES_FRAGMENTO = 900   # recorte por fragmento en la búsqueda: el agente puede leer el punto entero
MAX_CARACTERES_PUNTO = 6000      # techo al leer un punto completo, para no inflar el contexto

DocId = Literal["t-pusf", "t-depaho", "t-tasint", "t-efemin"]
Categoria = Literal["proteccion_usuarios", "cuentas_y_depositos", "tasas_y_credito", "regulacion_monetaria"]

# --- Sistema RAG compartido -------------------------------------------------------------------------

_sistema: RAGSystem | None = None
_candado = asyncio.Lock()


async def obtener_sistema() -> RAGSystem:
    """Crea el RAGSystem una sola vez (carga el corpus BM25 desde Pinecone) y lo reutiliza."""
    global _sistema
    async with _candado:
        if _sistema is None:
            _sistema = await asyncio.to_thread(RAGSystem)  # la inicialización es sincrónica (SDK Pinecone)
    return _sistema


def a_json(datos: dict[str, Any]) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=1)


def numero_de_punto(punto: str) -> str:
    """'2.6. Comisiones.' -> '2.6' | '2.6.' -> '2.6' | ' 2.6 ' -> '2.6'"""
    m = re.match(r"\s*(\d+(?:\.\d+)*)", punto)
    return m.group(1) if m else punto.strip().rstrip(".")


def resumir(doc: Document) -> dict[str, Any]:
    m = doc.metadata
    texto = doc.page_content
    return {
        "chunk_id": doc.id,
        "doc_id": m["doc_id"],
        "documento": m["documento"],
        "seccion": m["seccion"],
        "puntos": [numero_de_punto(p) for p in m["puntos"]],
        "coseno": m.get("puntaje_vectorial"),
        "texto": texto if len(texto) <= MAX_CARACTERES_FRAGMENTO else texto[:MAX_CARACTERES_FRAGMENTO] + " [...]",
    }


# --- Herramientas -----------------------------------------------------------------------------------

@tool
async def buscar_normativa(consulta: str, categoria: Categoria | None = None) -> str:
    """Busca fragmentos de la normativa del Banco Central de la República Argentina (BCRA) relevantes
    para una consulta en lenguaje natural. Combina búsqueda léxica (BM25) y semántica (embeddings).

    Usala SIEMPRE antes de responder cualquier pregunta sobre regulación bancaria argentina: no respondas
    de memoria. La base contiene solo estos 4 textos ordenados del BCRA:
      - t-pusf  (categoria "proteccion_usuarios"): Protección de los usuarios de servicios financieros
        (comisiones, información al cliente, publicidad, reclamos, rescisión, CFT informado al usuario).
      - t-depaho (categoria "cuentas_y_depositos"): Depósitos de ahorro, cuenta sueldo y especiales
        (caja de ahorros, cuenta sueldo, cuenta gratuita universal, tarjetas de débito, movimientos sin costo).
      - t-tasint (categoria "tasas_y_credito"): Tasas de interés en las operaciones de crédito
        (tasas, costo financiero total CFT, cálculo de intereses, préstamos y financiaciones).
      - t-efemin (categoria "regulacion_monetaria"): Efectivo mínimo (encajes, exigencia e integración).

    Para preguntas que combinan dos temas (ej. "comparar X con Y"), hacé una búsqueda por cada tema en
    lugar de una sola consulta mezclada.

    Args:
        consulta: la pregunta o los términos a buscar, en español. Conviene usar el vocabulario de la
            norma (ej. "costo financiero total" en vez de "cuánto me cobran de más").
        categoria: opcional. Restringe la búsqueda a un solo texto ordenado. Usala solo si el tema es
            claramente de esa norma; si la búsqueda sin filtro trae resultados de otras normas, no hace falta.

    Returns:
        JSON con "estado":
          - "ok": "resultados" trae hasta 5 fragmentos con chunk_id, doc_id, sección, números de punto,
            coseno (relevancia 0-1) y texto (recortado). Para leer un punto completo usá `leer_punto`.
          - "sin_resultados_relevantes": ningún fragmento supera el umbral de relevancia. Reformulá la
            consulta con otros términos (o sacá/cambiá la categoría) y volvé a buscar UNA vez más; si sigue
            sin resultados, decile al usuario que la normativa disponible no cubre el tema.
    """
    sistema = await obtener_sistema()
    docs = await sistema.buscar(consulta)
    if categoria:
        docs = [d for d in docs if d.metadata["categoria"] == categoria]

    mejor = max((d.metadata.get("puntaje_vectorial", 0.0) for d in docs), default=0.0)
    logger.info("buscar_normativa(%r, categoria=%s) -> %d docs, mejor coseno %.3f", consulta, categoria,
                len(docs), mejor)
    if mejor < UMBRAL_RELEVANCIA:
        return a_json({
            "estado": "sin_resultados_relevantes",
            "consulta": consulta,
            "categoria": categoria,
            "mejor_coseno": round(mejor, 3),
            "umbral": UMBRAL_RELEVANCIA,
            "sugerencia": "Reformulá con términos de la norma, probá sin categoría o con otra. Si tras un "
                          "segundo intento no hay resultados, el tema no está en la normativa disponible.",
        })
    return a_json({"estado": "ok", "consulta": consulta, "resultados": [resumir(d) for d in docs]})


@tool
async def leer_punto(doc_id: DocId, punto: str) -> str:
    """Devuelve el texto COMPLETO y textual de un punto de un texto ordenado del BCRA, buscándolo por
    clave exacta (doc_id + número de punto), como una consulta a base de datos.

    Usala cuando ya sabés qué punto necesitás (normalmente porque apareció en `buscar_normativa`) y el
    fragmento recortado no alcanza: por ejemplo para citar requisitos, listas completas, plazos o montos.
    No sirve para buscar por tema: para eso está `buscar_normativa`.

    Args:
        doc_id: identificador del texto ordenado: "t-pusf" (protección de usuarios), "t-depaho"
            (depósitos de ahorro y cuenta sueldo), "t-tasint" (tasas de interés en créditos) o
            "t-efemin" (efectivo mínimo).
        punto: número de punto con formato "sección.punto", ej. "2.6" o "1.8". Los puntos están
            indexados con dos niveles: si te piden "2.3.1.1", pedí "2.3" (el subpunto está adentro).

    Returns:
        JSON con "estado":
          - "ok": "texto" con el contenido del punto (todos sus fragmentos en orden), sección y chunk_ids.
          - "punto_inexistente": ese punto no existe en ese documento. "puntos_disponibles" lista los
            puntos válidos de la misma sección (o del documento): elegí uno de ahí y reintentá, o usá
            `buscar_normativa` si no sabés cuál corresponde.
    """
    sistema = await obtener_sistema()
    buscado = numero_de_punto(punto)
    corpus = [d for d in sistema.bm25.docs if d.metadata["doc_id"] == doc_id]
    chunks = [d for d in corpus if buscado in {numero_de_punto(p) for p in d.metadata["puntos"]}]

    logger.info("leer_punto(%s, %r) -> %d chunks", doc_id, punto, len(chunks))
    if not chunks:
        disponibles = sorted({numero_de_punto(p) for d in corpus for p in d.metadata["puntos"]},
                             key=lambda n: [int(x) for x in n.split(".") if x.isdigit()])
        seccion = buscado.split(".")[0] + "."
        de_la_seccion = [n for n in disponibles if n.startswith(seccion)]
        return a_json({
            "estado": "punto_inexistente",
            "doc_id": doc_id,
            "punto": buscado,
            "puntos_disponibles": de_la_seccion or disponibles,
            "sugerencia": "Elegí un punto de 'puntos_disponibles' o usá buscar_normativa para ubicar el tema.",
        })

    chunks.sort(key=lambda d: d.id)
    texto = "\n\n".join(d.page_content for d in chunks)
    m = chunks[0].metadata
    return a_json({
        "estado": "ok",
        "doc_id": doc_id,
        "documento": m["documento"],
        "seccion": m["seccion"],
        "punto": buscado,
        "chunk_ids": [d.id for d in chunks],
        "vigencia": m["vigencia"],
        "texto": texto if len(texto) <= MAX_CARACTERES_PUNTO else texto[:MAX_CARACTERES_PUNTO] + " [...]",
    })


HERRAMIENTAS = [buscar_normativa, leer_punto]


# --- Prueba aislada -----------------------------------------------------------------------------------

async def main() -> None:
    casos: list[tuple[Any, dict[str, Any]]] = [
        (buscar_normativa, {"consulta": "comisiones por mantenimiento de caja de ahorros"}),
        (buscar_normativa, {"consulta": "¿Cuánto cuesta un pasaje a Bariloche?"}),
        (leer_punto, {"doc_id": "t-depaho", "punto": "2.6"}),
        (leer_punto, {"doc_id": "t-depaho", "punto": "2.99"}),
    ]
    for herramienta, argumentos in casos:
        salida = json.loads(await herramienta.ainvoke(argumentos))
        print(f"\n=== {herramienta.name}({argumentos}) -> estado: {salida['estado']}")
        if "resultados" in salida:
            for r in salida["resultados"]:
                print(f"  {r['chunk_id']:<15} coseno {r['coseno']:.3f} | puntos {r['puntos']}")
        elif "texto" in salida:
            print(f"  {salida['seccion']} | chunks {salida['chunk_ids']}\n  {salida['texto'][:300]}...")
        else:
            print("  " + json.dumps({k: v for k, v in salida.items() if k != "estado"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
