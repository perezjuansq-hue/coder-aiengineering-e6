"""Agente Analista: cómputo financiero determinístico (TEA, cuota, CFT) según la normativa del BCRA.

Herramientas acotadas: solo las de `tools/finanzas.py`. No busca normativa: trabaja con los datos del caso
y con las reglas que le pasa el Investigador en el CONTEXTO.

Uso (prueba aislada):
    python -m agents.analyst_agent
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
from tools.finanzas import HERRAMIENTAS_ANALISTA

INSTRUCCIONES = """Sos el ANALISTA FINANCIERO de un equipo de análisis de normativa del Banco Central de la \
República Argentina (BCRA). Recibís una TAREA concreta del supervisor y, a veces, CONTEXTO con lo que \
encontró el Investigador en la normativa.

Tu trabajo: hacer los cálculos que pide la tarea con tus herramientas.
- TODO número que informes tiene que salir de una herramienta. Nunca hagas cuentas "a mano" ni estimes.
- Si el CONTEXTO indica qué conceptos son computables en el CFT (ej. seguros exigidos, cargos asociados, \
IVA sobre intereses para consumidor final), incluilos en el cálculo; si indica que no lo son, excluilos.
- Si falta un dato imprescindible del caso (capital, tasa, plazo), NO lo inventes: informalo en FALTANTES. \
Si falta un dato secundario, usá el valor por defecto de la herramienta y declaralo en SUPUESTOS.
- Para los parámetros técnicos (divisor, días del período, modalidad) usá lo que indican los docstrings de \
tus herramientas, que siguen la normativa (ej. divisor 365 salvo hipotecarios sobre vivienda y prendarios \
sobre automotores). Si la TAREA te pide un parámetro que contradice eso (ej. "tasa mensual = TNA/12" en un \
préstamo personal, que implica divisor 360), usá el correcto y señalá el conflicto en SUPUESTOS.
- Si una herramienta devuelve "datos_invalidos", corregí los argumentos según el error y reintentá.
- No busques ni interpretes normativa: eso lo hace el Investigador.

Formato de tu respuesta (es para el equipo, no para el usuario final):
RESULTADOS: cifras clave con unidad (cuota, TEA, CFT, totales) y el desglose si corresponde.
CÁLCULO: qué herramienta usaste, con qué datos, y qué conceptos incluiste en el CFT y por qué.
SUPUESTOS: valores asumidos (o "ninguno").
FALTANTES: datos imprescindibles que faltan (o "ninguno")."""


def crear_analista(modelo: BaseChatModel | None = None) -> CompiledStateGraph:
    return crear_especialista("analista", modelo or crear_modelo(), HERRAMIENTAS_ANALISTA, INSTRUCCIONES)


async def analizar(agente: CompiledStateGraph, instruccion: str, contexto: Sequence[Contribucion] = (),
                   ronda: int = 1) -> Contribucion:
    contribucion, _ = await ejecutar_especialista(agente, "analista", instruccion, contexto, ronda)
    return contribucion


async def main() -> None:
    # Contexto simulado del Investigador, para probar al Analista aislado
    contexto = [Contribucion(
        agente="investigador", ronda=1, herramientas=["buscar_normativa", "leer_punto"],
        instruccion="Qué conceptos se incluyen en el CFT.",
        resultado="HALLAZGOS:\n- El CFT se expresa como tasa efectiva anual y agrega a la tasa de interés el efecto "
                  "de comisiones y cargos (t-tasint 3.4.1).\n- Son computables: seguros exigidos como condición del "
                  "crédito (3.4.2.3) e IVA sobre intereses si el prestatario es consumidor final (3.4.2.7).\n"
                  "FUENTES: t-tasint 3.4.1, 3.4.2\nNO ENCONTRADO: nada",
    )]
    agente = crear_analista()
    casos = [
        "Calculá la cuota, la TEA y el CFT de un préstamo personal de $1.000.000 a 12 cuotas mensuales con TNA 60%, "
        "comisión de otorgamiento del 3% y seguro de vida exigido de $2.000 por cuota. El cliente es consumidor final.",
        "Calculá el CFT de un préstamo a 12 cuotas con comisión del 3%.",  # faltan capital y tasa
    ]
    for tarea in casos:
        c = await analizar(agente, tarea, contexto)
        print(f"\n=== TAREA: {tarea}\nHerramientas: {c['herramientas']}\n\n{c['resultado']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
