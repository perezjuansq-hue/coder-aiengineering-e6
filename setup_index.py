"""Inicialización de Pinecone: verifica que el índice serverless exista y, si no, lo crea.

Uso:
    python setup_index.py
"""
import logging
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("setup_index")

# La dimensión tiene que coincidir con el modelo de embeddings (text-embedding-3-small -> 1536).
# Si no coincide, Pinecone rechaza los upserts: es el "mismatch de dimensiones" de la consigna.
DIMENSION = 1536
METRICA = "cosine"
# El plan gratuito (Starter) solo permite serverless en AWS us-east-1
CLOUD = "aws"
REGION = "us-east-1"
# Un namespace por tipo de dato: hoy solo normativa, mañana podría haber otro (ej. FAQs de un banco)
NAMESPACE = "normativa-bcra"


def nombre_indice() -> str:
    load_dotenv()
    nombre = os.getenv("INDEX_NAME", "").strip()
    if not nombre:
        raise ValueError("Falta INDEX_NAME en el .env")
    return nombre


def cliente_pinecone() -> Pinecone:
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("Falta PINECONE_API_KEY en el .env")
    return Pinecone()  # lee PINECONE_API_KEY del entorno


def asegurar_indice(pc: Pinecone | None = None) -> str:
    """Crea el índice si no existe; si existe, verifica que su configuración sea compatible."""
    pc = pc or cliente_pinecone()
    nombre = nombre_indice()

    if not pc.has_index(nombre):
        logger.info("El índice '%s' no existe -> creándolo (serverless %s/%s, dim=%d, %s)",
                    nombre, CLOUD, REGION, DIMENSION, METRICA)
        pc.create_index(
            name=nombre,
            dimension=DIMENSION,
            metric=METRICA,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            tags={"proyecto": "coder-aiengineering-e4"},
        )  # bloquea hasta que el índice está listo
    else:
        logger.info("El índice '%s' ya existe", nombre)

    info = pc.describe_index(nombre)
    if info.dimension != DIMENSION or info.metric != METRICA:
        raise ValueError(
            f"El índice '{nombre}' tiene dim={info.dimension}, metric={info.metric}; se esperaba "
            f"dim={DIMENSION}, metric={METRICA}. Borralo o usá otro INDEX_NAME."
        )
    logger.info("Índice listo: dim=%d | metric=%s | host=%s", info.dimension, info.metric, info.host)
    return nombre


if __name__ == "__main__":
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    pc = cliente_pinecone()
    nombre = asegurar_indice(pc)
    stats = pc.Index(nombre).describe_index_stats()
    logger.info("Vectores totales: %d | namespaces: %s", stats.total_vector_count,
                {ns: v.vector_count for ns, v in stats.namespaces.items()} or "ninguno")
