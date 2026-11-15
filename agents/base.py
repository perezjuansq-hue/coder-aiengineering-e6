"""Piezas comunes de los especialistas: construcción con create_agent y ejecución aislada.

Cada especialista es un agente ReAct propio (modelo ⇄ herramientas) creado con `create_agent`, el
reemplazo oficial de `create_react_agent` (deprecado en LangGraph 1.x). El supervisor no le pasa la
conversación entera: solo una instrucción concreta y los resultados previos de otros especialistas que
necesita ("contaminación de contexto"). El especialista devuelve una `Contribucion` atribuida.
"""
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallLimitMiddleware, wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from state import Contribucion, Especialista

logger = logging.getLogger("agents")

# Techo de llamadas a herramientas por tarea. Al llegar, las llamadas extra se BLOQUEAN (vuelven como error)
# y el especialista tiene que redactar con lo que juntó: no se pierde el trabajo hecho.
MAX_LLAMADAS_HERRAMIENTAS = 8
# Techo de pasos del grafo interno. Cada vuelta usa 3 nodos (model -> ToolCallLimitMiddleware.after_model ->
# tools): 8 llamadas (aun sin paralelismo) + vuelta de cierre ≈ 30 pasos; 40 deja margen.
RECURSION_LIMIT_ESPECIALISTA = 40


@wrap_model_call
async def consultar_antes_de_responder(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Guardrail (como en la PE5): mientras el especialista no haya usado ninguna herramienta, está obligado
    a llamar una ("any" = "required" en OpenAI). Después elige libremente; si lo forzáramos siempre, nunca
    podría terminar."""
    ya_uso_herramientas = any(isinstance(m, ToolMessage) for m in request.messages)
    if not ya_uso_herramientas:
        request = request.override(tool_choice="any")
    return await handler(request)


def crear_especialista(nombre: Especialista, modelo: BaseChatModel, herramientas: Sequence[BaseTool],
                       instrucciones: str) -> CompiledStateGraph:
    return create_agent(
        modelo,
        tools=list(herramientas),
        system_prompt=instrucciones,
        middleware=[consultar_antes_de_responder,
                    # "continue": bloquea las llamadas que exceden el límite y deja que el modelo cierre.
                    # Con "end" el agente terminaba sin redactar y se perdía todo lo investigado.
                    ToolCallLimitMiddleware(run_limit=MAX_LLAMADAS_HERRAMIENTAS, exit_behavior="continue")],
        name=nombre,
    )


def armar_mensaje(instruccion: str, contexto: Sequence[Contribucion]) -> str:
    """Lo único que ve el especialista: su tarea + los resultados (no las trazas) de otros especialistas."""
    partes = [f"TAREA:\n{instruccion}"]
    if contexto:
        aportes = "\n\n".join(f"[{c['agente']} · ronda {c['ronda']}] Tarea: {c['instruccion']}\nResultado:\n{c['resultado']}"
                              for c in contexto)
        partes.append(f"CONTEXTO (resultados de otros especialistas del equipo):\n{aportes}")
    return "\n\n".join(partes)


async def ejecutar_especialista(agente: CompiledStateGraph, nombre: Especialista, instruccion: str,
                                contexto: Sequence[Contribucion], ronda: int) -> tuple[Contribucion, list[Any]]:
    """Corre el especialista en un hilo efímero (sin memoria propia) y devuelve su contribución y los
    mensajes internos (para la traza). Nunca lanza: si se pasa del límite, lo informa como resultado."""
    entrada = {"messages": [HumanMessage(armar_mensaje(instruccion, contexto))]}
    mensajes: list[Any] = entrada["messages"]
    cortado = False
    try:
        # stream de "values": si se corta por recursion_limit, conservamos el último estado (no se pierde
        # qué herramientas usó ni lo que encontró)
        async for valores in agente.astream(entrada, {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA},
                                            stream_mode="values"):
            mensajes = valores["messages"]
    except GraphRecursionError:
        cortado = True

    resultado = next((m.text for m in reversed(mensajes) if isinstance(m, AIMessage) and not m.tool_calls
                      and m.text.strip()), "")
    if cortado or not resultado:
        observaciones = [m.text[:600] for m in mensajes if isinstance(m, ToolMessage)][-3:]
        resultado = (f"(Tarea incompleta: el especialista no llegó a redactar su resultado en "
                     f"{RECURSION_LIMIT_ESPECIALISTA} pasos. Últimos datos obtenidos por sus herramientas:)\n"
                     + "\n---\n".join(observaciones or ["ninguno"]))

    herramientas = [c["name"] for m in mensajes if isinstance(m, AIMessage) for c in m.tool_calls]
    logger.info("%s (ronda %d) usó %s", nombre, ronda, herramientas)
    contribucion = Contribucion(agente=nombre, ronda=ronda, instruccion=instruccion, resultado=resultado,
                                herramientas=herramientas)
    return contribucion, mensajes
