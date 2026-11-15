"""Supervisor: router inteligente y controlador de flujo del equipo.

En cada vuelta mira el pedido del usuario y las contribuciones acumuladas (no las trazas internas de los
especialistas), aplica una rúbrica de validación y decide con salida estructurada:
    destino ∈ {"investigador", "analista", "sintesis"} + instrucción concreta + motivo.

Condiciones de parada (anti "supervisor infinito"):
    1. Criterio de suficiencia: la rúbrica se cumple -> "sintesis".
    2. Contador duro: al llegar a MAX_DELEGACIONES se va a "sintesis" sin consultar al LLM.
"""
import logging
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import Contribucion, DecisionSupervisor

logger = logging.getLogger("supervisor")

MAX_DELEGACIONES = 5

INSTRUCCIONES = """Sos el SUPERVISOR de un equipo que responde consultas sobre normativa del Banco Central \
de la República Argentina (BCRA). No respondés vos: decidís quién trabaja y validás el trabajo.

EQUIPO
- "investigador": busca y lee la normativa del BCRA (protección de usuarios, cuentas y depósitos, tasas y \
crédito, efectivo mínimo). NO calcula.
- "analista": hace cálculos financieros con herramientas (TNA→TEA, cuota, CFT de un préstamo, CFT de flujos). \
NO busca normativa.
- "sintesis": redacta la respuesta final al usuario con las contribuciones. Elegila solo cuando se cumple la rúbrica.

CÓMO DELEGAR
- La instrucción tiene que ser concreta y AUTOCONTENIDA: el especialista NO ve la conversación con el \
usuario. Al analista pasale todos los datos numéricos del caso; al investigador, qué tiene que averiguar.
- NO agregues supuestos técnicos propios (fórmulas, divisor de días, tasa mensual = TNA/12, sistema de \
amortización, etc.): pasá solo los datos que dio el usuario y lo que encontró el investigador. Los \
parámetros de cálculo los define el analista con sus herramientas, que siguen la normativa.
- Si el pedido necesita normativa Y cálculos, primero el investigador (para que el analista sepa qué \
conceptos computar) y después el analista.
- Saludos o preguntas ajenas a la normativa del BCRA: "sintesis" directo, sin delegar.

RÚBRICA (evaluala sobre las contribuciones antes de elegir "sintesis"; tienen que cumplirse todas):
1. COBERTURA: cada parte del pedido del usuario está respondida por alguna contribución.
2. FUNDAMENTO: las afirmaciones sobre la norma vienen del investigador y citan fuente (doc_id y punto).
3. CIFRAS: todo número del resultado viene del analista y el analista usó una herramienta de cálculo \
(mirá su lista de herramientas). Si informó FALTANTES, esos datos no están en el pedido del usuario.
4. COHERENCIA: los conceptos que el analista incluyó o excluyó del cálculo coinciden con lo que el \
investigador encontró en la norma.
5. SUPUESTOS: cada supuesto del analista que no dio el usuario es el valor por defecto de su herramienta \
o está respaldado por una fuente del investigador. Si el analista se apartó de los valores por defecto sin \
respaldo, refinalo.

SI ALGO FALLA
- Devolvé el trabajo al especialista que corresponda con una instrucción de REFINAMIENTO que diga \
exactamente qué corregir o completar (ej. "Recalculá incluyendo el seguro exigido, que según t-tasint \
3.4.2.3 es computable").
- CONFLICTOS: si el investigador y el analista no coinciden, la norma (investigador) manda sobre el \
cálculo: refiná al analista. Si el conflicto es sobre qué dice la norma, refiná al investigador.
- No repitas una instrucción que ya se dio con el mismo resultado. Si un especialista ya informó que algo \
NO está en la normativa o que falta un dato que el usuario no dio, no insistas: andá a "sintesis" y la \
síntesis lo va a aclarar."""


class Decision(BaseModel):
    """Decisión de ruteo del supervisor."""
    evaluacion: str = Field(description="Evaluación breve de las contribuciones contra los 5 criterios de la "
                                        "rúbrica (o 'sin contribuciones todavía').")
    destino: Literal["investigador", "analista", "sintesis"] = Field(description="Quién interviene ahora.")
    instruccion: str = Field(description="Tarea concreta y autocontenida para el especialista. Vacía si el "
                                         "destino es 'sintesis'.")
    motivo: str = Field(description="Por qué elegiste ese destino, en una oración.")


def describir_contribuciones(contribuciones: Sequence[Contribucion]) -> str:
    if not contribuciones:
        return "(todavía no hay contribuciones)"
    return "\n\n".join(
        f"[{c['agente']} · ronda {c['ronda']}] herramientas usadas: {c['herramientas'] or 'ninguna'}\n"
        f"Tarea: {c['instruccion']}\nResultado:\n{c['resultado']}"
        for c in contribuciones
    )


async def decidir(modelo: BaseChatModel, pedido: str, contribuciones: Sequence[Contribucion],
                  decisiones: Sequence[DecisionSupervisor], delegaciones: int) -> tuple[DecisionSupervisor, str]:
    """Devuelve la decisión (para el estado) y la evaluación de la rúbrica (para la traza)."""
    ronda = delegaciones + 1
    if delegaciones >= MAX_DELEGACIONES:
        motivo = f"Se alcanzó el máximo de {MAX_DELEGACIONES} delegaciones: se cierra con lo disponible."
        logger.warning(motivo)
        return DecisionSupervisor(ronda=ronda, destino="sintesis", instruccion="", motivo=motivo), "límite alcanzado"

    historial = "\n".join(f"- ronda {d['ronda']} → {d['destino']}: {d['instruccion'] or '(fin)'}" for d in decisiones)
    entrada = (f"PEDIDO DEL USUARIO:\n{pedido}\n\n"
               f"DELEGACIONES REALIZADAS: {delegaciones} de {MAX_DELEGACIONES}\n{historial}\n\n"
               f"CONTRIBUCIONES:\n{describir_contribuciones(contribuciones)}")
    decision: Decision = await modelo.with_structured_output(Decision).ainvoke(
        [SystemMessage(INSTRUCCIONES), HumanMessage(entrada)])

    instruccion = decision.instruccion.strip()
    destino = decision.destino
    if destino != "sintesis" and not instruccion:
        # Delegar sin tarea no sirve: el especialista no ve el pedido. Le pasamos el pedido completo.
        instruccion = pedido
    return (DecisionSupervisor(ronda=ronda, destino=destino,
                               instruccion=instruccion if destino != "sintesis" else "",
                               motivo=decision.motivo), decision.evaluacion)
