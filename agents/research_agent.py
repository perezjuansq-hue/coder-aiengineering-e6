"""Agente Investigador: busca y lee la normativa del BCRA (RAG híbrido de la PE4/PE5).

Herramientas acotadas: solo `buscar_normativa` y `leer_punto`. No calcula: si la tarea pide números,
trae las reglas y definiciones que el Analista necesita para calcular.

Uso (prueba aislada):
    python -m agents.research_agent "¿Qué conceptos se incluyen en el CFT y cómo debe expresarse?"
"""
import asyncio
import logging
import sys
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agents.base import crear_especialista, ejecutar_especialista
from llm import crear_modelo
from state import Contribucion
from tools.normativa import buscar_normativa, leer_punto

INSTRUCCIONES = """Sos el INVESTIGADOR de un equipo de análisis de normativa del Banco Central de la \
República Argentina (BCRA). Recibís una TAREA concreta del supervisor del equipo y, a veces, CONTEXTO con \
resultados de otros especialistas.

Tu trabajo: encontrar en la normativa lo que la tarea pide, usando tus herramientas.
- Usá `buscar_normativa` para ubicar el tema y `leer_punto` para leer completo el punto relevante cuando \
el fragmento no alcance (definiciones, listas de conceptos, requisitos).
- Sé eficiente: tenés un máximo de 8 llamadas por tarea. Primero buscá; después leé SOLO los puntos que \
responden la tarea (no leas puntos "por las dudas"). Si una herramienta te avisa que llegaste al límite, \
redactá tu respuesta con lo que ya tenés.
- Si una herramienta devuelve un error o resultados no relevantes, seguí su "sugerencia" y reintentá una vez.
- Basate SOLO en lo que devuelven las herramientas. No completes con conocimiento propio.
- NO hagas cálculos numéricos: otro especialista (el Analista) se encarga. Si la tarea involucra números, \
traé las reglas, definiciones y conceptos que hacen falta para calcular.

Formato de tu respuesta (es para el equipo, no para el usuario final):
HALLAZGOS: puntos concretos y textuales de la norma que responden la tarea, en viñetas.
FUENTES: doc_id y número de punto de cada hallazgo (ej. t-tasint 3.4.2).
NO ENCONTRADO: lo que la tarea pedía y no está en la normativa disponible (o "nada")."""


def crear_investigador(modelo: BaseChatModel | None = None) -> CompiledStateGraph:
    return crear_especialista("investigador", modelo or crear_modelo(), [buscar_normativa, leer_punto], INSTRUCCIONES)


async def investigar(agente: CompiledStateGraph, instruccion: str, contexto: Sequence[Contribucion] = (),
                     ronda: int = 1) -> Contribucion:
    contribucion, _ = await ejecutar_especialista(agente, "investigador", instruccion, contexto, ronda)
    return contribucion


async def main() -> None:
    tarea = " ".join(sys.argv[1:]) or ("Averiguá qué conceptos se incluyen y cuáles no en el costo financiero "
                                       "total (CFT) y cómo debe expresarse según la normativa.")
    c = await investigar(crear_investigador(), tarea)
    print(f"Herramientas: {c['herramientas']}\n\n{c['resultado']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
