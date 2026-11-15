"""Grafo del orquestador multi-agente (topología jerárquica: supervisor + especialistas + síntesis).

    START ──► supervisor ──(rutear: Literal)──► investigador ──┐
                 ▲    │                   └──► analista ─────┤
                 │    └──► sintesis ──► END                  │
                 └───────────────────────────────────────────┘
    Los especialistas siempre devuelven el control al supervisor, que valida con su rúbrica y decide si
    refinar, delegar a otro o cerrar.

Uso:
    python graph.py                       # consulta de prueba (préstamo: normativa + cálculo)
    python graph.py "tu consulta"
    python graph.py --mermaid             # imprime el diagrama Mermaid del grafo
"""
import asyncio
import logging
import sys
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import crear_analista
from agents.base import ejecutar_especialista
from agents.research_agent import crear_investigador
from agents.supervisor import MAX_DELEGACIONES, decidir, describir_contribuciones
from llm import crear_modelo
from state import Destino, EstadoOrquestador

logger = logging.getLogger("graph")

# Pasos del grafo externo: cada delegación usa 2 (especialista + supervisor), más inicio y síntesis.
# Con MAX_DELEGACIONES = 5 alcanzan ~13; 25 es el techo de seguridad por si algo se sale de lo previsto.
RECURSION_LIMIT = 25

CONSULTA_DEMO = ("Me ofrecen un préstamo personal de $1.000.000 a 12 cuotas mensuales con TNA del 60%, una "
                 "comisión de otorgamiento del 3% y un seguro de vida obligatorio de $2.000 por cuota. Soy "
                 "consumidor final. ¿Qué conceptos tienen que incluir en el CFT según el BCRA, cuánto me da el "
                 "CFT y cómo me lo tienen que informar?")

INSTRUCCIONES_SINTESIS = """Sos el redactor final de un equipo que responde consultas sobre normativa del \
BCRA. Recibís el pedido del usuario y las contribuciones validadas de los especialistas.

- Respondé al usuario en español rioplatense, claro y ordenado (podés usar viñetas y negritas).
- Usá SOLO la información de las contribuciones: no agregues datos, cifras ni normas que no estén ahí.
- Copiá las cifras tal cual las informó el analista.
- Citá las fuentes normativas (doc_id y punto) que dio el investigador.
- Si hay supuestos, faltantes o partes que la normativa no cubre, decilo explícitamente.
- Si no hay contribuciones (saludo o tema ajeno), presentate como asistente de normativa del BCRA y contá \
qué temas cubrís: protección de usuarios, cuentas y depósitos, tasas y crédito, efectivo mínimo."""


def pedido_actual(estado: EstadoOrquestador) -> str:
    return next(m.text for m in reversed(estado["messages"]) if isinstance(m, HumanMessage))


def construir_grafo(modelo: BaseChatModel | None = None) -> CompiledStateGraph:
    modelo = modelo or crear_modelo()
    investigador = crear_investigador(modelo)
    analista = crear_analista(modelo)

    async def supervisor(estado: EstadoOrquestador) -> dict[str, Any]:
        decision, evaluacion = await decidir(modelo, pedido_actual(estado), estado.get("contribuciones", []),
                                             estado.get("decisiones", []), estado.get("delegaciones", 0))
        logger.info("supervisor ronda %d -> %s | %s", decision["ronda"], decision["destino"], evaluacion)
        cambios: dict[str, Any] = {"decisiones": [decision], "siguiente": decision["destino"]}
        if decision["destino"] == "sintesis":
            # Completa si el supervisor la validó; si se cortó por el contador, queda en False
            cambios["tarea_completa"] = estado.get("delegaciones", 0) < MAX_DELEGACIONES
        else:
            cambios["instruccion"] = decision["instruccion"]
            cambios["delegaciones"] = estado.get("delegaciones", 0) + 1
        return cambios

    async def nodo_investigador(estado: EstadoOrquestador) -> dict[str, Any]:
        contribucion, _ = await ejecutar_especialista(investigador, "investigador", estado["instruccion"],
                                                      estado.get("contribuciones", []), estado["delegaciones"])
        return {"contribuciones": [contribucion]}

    async def nodo_analista(estado: EstadoOrquestador) -> dict[str, Any]:
        contribucion, _ = await ejecutar_especialista(analista, "analista", estado["instruccion"],
                                                      estado.get("contribuciones", []), estado["delegaciones"])
        return {"contribuciones": [contribucion]}

    async def sintesis(estado: EstadoOrquestador) -> dict[str, Any]:
        aviso = "" if estado.get("tarea_completa", True) else (
            "\n\nATENCIÓN: el equipo llegó al límite de delegaciones sin validar todo. Aclaralo en la respuesta.")
        entrada = (f"PEDIDO DEL USUARIO:\n{pedido_actual(estado)}\n\n"
                   f"CONTRIBUCIONES:\n{describir_contribuciones(estado.get('contribuciones', []))}{aviso}")
        respuesta = await modelo.ainvoke([SystemMessage(INSTRUCCIONES_SINTESIS), HumanMessage(entrada)])
        return {"messages": [AIMessage(respuesta.text, name="sintesis")]}

    def rutear(estado: EstadoOrquestador) -> Literal["investigador", "analista", "sintesis"]:
        """Arista condicional: mapea la decisión del supervisor al nombre del nodo."""
        destino: Destino = estado["siguiente"]
        return destino

    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("supervisor", supervisor)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("sintesis", sintesis)
    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", rutear, ["investigador", "analista", "sintesis"])
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("sintesis", END)
    return grafo.compile()


def estado_inicial(consulta: str) -> dict[str, Any]:
    return {"messages": [HumanMessage(consulta)], "contribuciones": [], "decisiones": [], "delegaciones": 0,
            "tarea_completa": False}


async def ejecutar(grafo: CompiledStateGraph, consulta: str, verbose: bool = True) -> dict[str, Any]:
    """Corre el orquestador mostrando la traza de delegación y devuelve el estado final."""
    if verbose:
        print(f'USUARIO: "{consulta}"\n')
    final: dict[str, Any] = {}
    async for evento in grafo.astream(estado_inicial(consulta), {"recursion_limit": RECURSION_LIMIT},
                                      stream_mode=["updates", "values"]):
        modo, datos = evento
        if modo == "values":
            final = datos
            continue
        if not verbose:
            continue
        for nodo, cambios in datos.items():
            if nodo == "supervisor":
                d = cambios["decisiones"][0]
                print(f"[supervisor · ronda {d['ronda']}] → {d['destino'].upper()}  ({d['motivo']})")
                if d["instruccion"]:
                    print(f"    instrucción: {d['instruccion']}")
            elif nodo in ("investigador", "analista"):
                c = cambios["contribuciones"][0]
                print(f"[{nodo}] herramientas: {', '.join(c['herramientas']) or 'ninguna'}")
                print("    " + c["resultado"][:400].replace("\n", "\n    ") + ("..." if len(c["resultado"]) > 400 else ""))
            elif nodo == "sintesis":
                print(f"\n[sintesis] RESPUESTA FINAL:\n{cambios['messages'][0].text}")
        print()
    return final


async def main() -> None:
    if "--mermaid" in sys.argv:
        print(construir_grafo().get_graph().draw_mermaid())
        return
    consulta = " ".join(a for a in sys.argv[1:]) or CONSULTA_DEMO
    final = await ejecutar(construir_grafo(), consulta)
    print(f"delegaciones: {final.get('delegaciones')} | tarea_completa: {final.get('tarea_completa')} | "
          f"contribuciones: {[(c['agente'], c['ronda']) for c in final.get('contribuciones', [])]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
