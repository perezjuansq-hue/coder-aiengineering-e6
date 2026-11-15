"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) combinados con un EnsembleRetriever.

Uso:
    python rag_system.py "¿Tu consulta?"   # muestra el top-5 de cada modo (bm25, vectorial, híbrido)
"""
import asyncio
import logging
import re
import sys
import unicodedata
import warnings
from typing import Any, List, Literal

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import Index

from ingest import abrir_indice, crear_embeddings, indexar
from setup_index import NAMESPACE

with warnings.catch_warnings():
    # langchain-community avisa que está siendo discontinuado; BM25Retriever todavía vive ahí
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

logger = logging.getLogger("rag_system")

TOP_K = 5
# Cada retriever trae más candidatos que el top final: la fusión necesita "material" de ambos lados
CANDIDATOS_POR_RETRIEVER = 10
# (bm25, vectorial). Con 0.5/0.5 el ruido léxico de BM25 empeoraba las preguntas parafraseadas;
# con 0.3/0.7 el híbrido iguala o supera al vectorial en todas las métricas (ver evaluate.py y README)
PESOS = (0.3, 0.7)
LOTE_FETCH = 100

Modo = Literal["bm25", "vectorial", "hibrido"]


# --- Tokenizador para BM25 ----------------------------------------------------------------------

STOPWORDS = {
    "a", "al", "ante", "con", "como", "cual", "cuales", "cuando", "de", "del", "desde", "donde", "e", "el",
    "ella", "ellos", "en", "entre", "es", "esa", "ese", "eso", "esta", "este", "esto", "fue", "ha", "hay",
    "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "muy", "ni", "no", "nos", "o", "para", "pero",
    "por", "que", "quien", "se", "ser", "si", "sin", "sobre", "son", "su", "sus", "te", "tu", "un", "una",
    "uno", "unos", "unas", "y", "ya", "yo",
}
# Números de punto de la norma ("2.3.1.1") como un solo token; si no, quedarían "2", "3", "1", "1"
PATRON_TOKEN = re.compile(r"\d+(?:\.\d+)+|\w+")


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> List[str]:
    """'¿Cuánto cobran por el CFT del punto 2.3.1.1.?' -> ['cuanto', 'cobran', 'cft', 'punto', '2.3.1.1']

    El tokenizador por defecto de BM25Retriever es texto.split(): "¿Cuánto" y "cuánto" serían palabras
    distintas, y los signos de puntuación quedarían pegados a las palabras.
    """
    tokens = PATRON_TOKEN.findall(sin_acentos(texto.lower()))
    return [t for t in tokens if t not in STOPWORDS]


# --- Retriever de Pinecone (SDK nativo) -----------------------------------------------------------

def a_documento(id_: str, metadata: dict, puntaje: float | None = None) -> Document:
    """Registro de Pinecone -> Document de LangChain. El texto viene de la metadata (campo "texto")."""
    meta = {k: v for k, v in metadata.items() if k != "texto"}
    if puntaje is not None:
        meta["puntaje_vectorial"] = puntaje
    return Document(id=id_, page_content=metadata["texto"], metadata=meta)


class PineconeRetriever(BaseRetriever):
    """Búsqueda por similitud coseno en un namespace de Pinecone.

    Equivale al PineconeVectorStore(...).as_retriever() de langchain-pinecone, que todavía no
    soporta Python 3.14: embebe la consulta, consulta el índice y arma los Document con la metadata.
    """

    indice: Any
    embeddings: Embeddings
    namespace: str = NAMESPACE
    k: int = CANDIDATOS_POR_RETRIEVER
    filtro: dict | None = None

    def _consultar(self, vector: List[float]) -> List[Document]:
        respuesta = self.indice.query(vector=vector, top_k=self.k, namespace=self.namespace,
                                      filter=self.filtro, include_metadata=True)
        return [a_documento(m.id, m.metadata, round(m.score, 4)) for m in respuesta.matches]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self._consultar(self.embeddings.embed_query(query))

    async def _aget_relevant_documents(self, query: str, *,
                                       run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        vector = await self.embeddings.aembed_query(query)
        return await asyncio.to_thread(self._consultar, vector)  # el SDK de Pinecone es sincrónico


# --- Sistema de recuperación ----------------------------------------------------------------------

def cargar_corpus(indice: Index, namespace: str = NAMESPACE) -> List[Document]:
    """Lee todos los chunks del namespace desde Pinecone para construir el índice BM25.

    Pinecone es la única fuente de verdad: BM25 y la búsqueda vectorial ven exactamente los mismos
    chunks. Funciona porque el texto original está guardado en la metadata de cada vector.
    """
    ids = [item.id for pagina in indice.list(namespace=namespace) for item in pagina.vectors]
    docs = []
    for i in range(0, len(ids), LOTE_FETCH):
        respuesta = indice.fetch(ids=ids[i:i + LOTE_FETCH], namespace=namespace)
        docs += [a_documento(id_, v.metadata) for id_, v in respuesta.vectors.items()]
    docs.sort(key=lambda d: d.id)
    return docs


class RAGSystem:
    """Encapsula el EnsembleRetriever: recibe una consulta y devuelve los top-5 chunks combinando
    resultados léxicos (BM25) y semánticos (Pinecone) con Reciprocal Rank Fusion ponderada."""

    def __init__(self, top_k: int = TOP_K, pesos: tuple[float, float] = PESOS, reindexar: bool = False):
        self.top_k = top_k
        self.indice = indexar(forzar=reindexar)  # reutiliza el índice si existe y está actualizado
        self.embeddings = crear_embeddings()

        corpus = cargar_corpus(self.indice)
        logger.info("Corpus BM25 cargado desde Pinecone: %d chunks", len(corpus))
        self.bm25 = BM25Retriever.from_documents(corpus, preprocess_func=tokenizar, k=CANDIDATOS_POR_RETRIEVER)
        self.vectorial = PineconeRetriever(indice=self.indice, embeddings=self.embeddings)
        self.hibrido = EnsembleRetriever(retrievers=[self.bm25, self.vectorial], weights=list(pesos),
                                         id_key="chunk_id")  # deduplica por id, no por texto

    async def buscar(self, consulta: str, modo: Modo = "hibrido") -> List[Document]:
        """Top-k documentos para la consulta. `modo` permite comparar cada retriever por separado."""
        retriever = {"bm25": self.bm25, "vectorial": self.vectorial, "hibrido": self.hibrido}[modo]
        # El Ensemble devuelve la unión de ambas listas (hasta 2*k): nos quedamos con los top_k fusionados
        docs = (await retriever.ainvoke(consulta))[: self.top_k]
        if modo != "bm25":
            await self._completar_puntajes(consulta, docs)
        return docs

    async def obtener_top_k(self, consulta: str) -> List[Document]:
        return await self.buscar(consulta, "hibrido")

    async def _completar_puntajes(self, consulta: str, docs: List[Document]) -> None:
        """Similitud coseno de cada documento del top-k con la consulta.

        Los que llegaron por BM25 no traen puntaje vectorial: se lo pedimos a Pinecone con una consulta
        filtrada a esos ids. Sirve para el umbral de relevancia y para mostrar la relevancia en las citas.
        """
        faltan = [d.metadata["chunk_id"] for d in docs if "puntaje_vectorial" not in d.metadata]
        if not faltan:
            return
        vector = await self.embeddings.aembed_query(consulta)
        respuesta = await asyncio.to_thread(
            self.indice.query, vector=vector, top_k=len(faltan), namespace=NAMESPACE,
            filter={"chunk_id": {"$in": faltan}},
        )
        puntajes = {m.id: round(m.score, 4) for m in respuesta.matches}
        for d in docs:
            d.metadata.setdefault("puntaje_vectorial", puntajes.get(d.metadata["chunk_id"], 0.0))


# --- Demo ---------------------------------------------------------------------------------------------

async def main() -> None:
    consulta = " ".join(sys.argv[1:]) or "¿Qué es el CFT y cómo se expresa?"
    sistema = RAGSystem()
    print(f"\nConsulta: {consulta}\nTokens BM25: {tokenizar(consulta)}")
    for modo in ("bm25", "vectorial", "hibrido"):
        print(f"\n--- {modo} ---")
        for i, d in enumerate(await sistema.buscar(consulta, modo), start=1):
            m = d.metadata
            coseno = f"{m['puntaje_vectorial']:.3f}" if "puntaje_vectorial" in m else "  -  "
            print(f"{i}. {d.id:<15} coseno {coseno} | {m['punto'][:70] or m['seccion'][:70]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for nombre in ("httpx", "httpx2", "openai", "pinecone"):
        logging.getLogger(nombre).setLevel(logging.WARNING)
    asyncio.run(main())
