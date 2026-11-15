"""Estado compartido del orquestador multi-agente.

Además de `messages` (la conversación con el usuario, heredada de MessagesState), el estado registra
QUIÉN aportó QUÉ: cada especialista agrega una `Contribucion` y cada decisión del supervisor queda en
`decisiones`. Así el supervisor (y la síntesis) trabajan sobre datos estructurados y atribuidos, no sobre
un historial de chat mezclado, y la traza de delegación queda en el propio estado.
"""
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

# Nombres de los nodos del grafo: el supervisor solo puede devolver uno de estos destinos
Especialista = Literal["investigador", "analista"]
Destino = Literal["investigador", "analista", "sintesis"]


class Contribucion(TypedDict):
    """Lo que un especialista aportó en una delegación."""
    agente: Especialista
    ronda: int                       # número de delegación en la que se produjo
    instruccion: str                 # la tarea concreta que le dio el supervisor
    resultado: str                   # su respuesta final (resumen de lo que encontró o calculó)
    herramientas: list[str]          # herramientas que usó, en orden (p. ej. ["buscar_normativa", "leer_punto"])


class DecisionSupervisor(TypedDict):
    """Una decisión de ruteo del supervisor, con su justificación (la rúbrica aplicada)."""
    ronda: int
    destino: Destino
    instruccion: str                 # vacía cuando el destino es "sintesis"
    motivo: str                      # por qué delega o por qué considera la tarea completa


class EstadoOrquestador(MessagesState):
    """Estado del grafo. Los campos con `operator.add` se ACUMULAN (cada nodo devuelve solo lo nuevo);
    el resto se sobrescribe con el último valor que devuelva un nodo."""
    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[DecisionSupervisor], operator.add]
    siguiente: Destino               # última decisión del supervisor (la lee la arista condicional)
    instruccion: str                 # tarea vigente para el especialista elegido
    delegaciones: int                # contador anti "supervisor infinito"
    tarea_completa: bool             # True cuando el supervisor da por suficiente el trabajo (o se forzó el cierre)
