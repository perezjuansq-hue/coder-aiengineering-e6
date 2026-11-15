"""Herramientas de cómputo financiero del Agente Analista, implementadas según la normativa del BCRA
"Tasas de interés en las operaciones de crédito" (t-tasint).

Las fórmulas del punto 3.3 son imágenes en el PDF original: no están en la base vectorial, así que el
LLM no las puede "leer" ni debería reconstruirlas de memoria. Por eso el cálculo es determinístico:
el LLM decide QUÉ calcular y con qué datos; la cuenta la hace Python.

    3.3.1 (intereses vencidos):    i = [ (1 + is · m / (df·100)) ^ (df/m) − 1 ] · 100
    3.3.2 (intereses adelantados): i = [ (1 / (1 − d · m / (df·100))) ^ (df/m) − 1 ] · 100
      i: tasa efectiva anual (%) · is: tasa nominal anual pactada (%) · d: tasa de descuento anual (%)
      m: días del subperíodo (30 si es mensual) · df: divisor fijo, 365 (o 360 en hipotecarios/prendarios)

    3.4.1 CFT: tasa efectiva anual que agrega a la tasa de interés el efecto de comisiones y cargos.
          Acá: TIR por período de los flujos reales del cliente, anualizada con la misma lógica de 3.3.1.

Uso (prueba aislada con casos verificables a mano):
    python -m tools.finanzas
"""
import json
import sys
from typing import Any, Literal

from langchain_core.tools import tool

IVA_GENERAL_PCT = 21.0
MAX_CUOTAS = 600


# --- Funciones puras ---------------------------------------------------------------------------------

def tea_desde_nominal(tasa_pct: float, dias_periodo: int = 30, divisor: int = 365,
                      modalidad: Literal["vencida", "adelantada"] = "vencida") -> float:
    """Tasa efectiva anual (%) según 3.3.1 (vencida) o 3.3.2 (adelantada)."""
    proporcional = tasa_pct * dias_periodo / (divisor * 100)
    base = 1 + proporcional if modalidad == "vencida" else 1 / (1 - proporcional)
    return (base ** (divisor / dias_periodo) - 1) * 100


def anualizar(tasa_periodo: float, dias_periodo: int = 30, divisor: int = 365) -> float:
    """Tasa efectiva por período (en tanto por uno) -> tasa efectiva anual (%), misma capitalización que 3.3.1."""
    return ((1 + tasa_periodo) ** (divisor / dias_periodo) - 1) * 100


def tir(desembolso: float, pagos: list[float], tolerancia: float = 1e-12) -> float | None:
    """Tasa por período que iguala el desembolso recibido con el valor actual de los pagos (bisección).

    Con pagos positivos el valor actual decrece con la tasa: hay una única raíz si existe.
    Devuelve None si no hay solución en (-99%, 1000%) por período.
    """
    def valor_neto(r: float) -> float:
        return sum(p / (1 + r) ** k for k, p in enumerate(pagos, start=1)) - desembolso

    bajo, alto = -0.99, 10.0
    if valor_neto(bajo) * valor_neto(alto) > 0:
        return None
    for _ in range(300):
        medio = (bajo + alto) / 2
        if valor_neto(medio) > 0:
            bajo = medio
        else:
            alto = medio
        if alto - bajo < tolerancia:
            break
    return (bajo + alto) / 2


def cuota_frances(capital: float, tasa_periodo: float, cuotas: int) -> float:
    if tasa_periodo == 0:
        return capital / cuotas
    return capital * tasa_periodo / (1 - (1 + tasa_periodo) ** -cuotas)


def simular(capital: float, tna_pct: float, cuotas: int, dias_periodo: int = 30, divisor: int = 365,
            comision_otorgamiento_pct: float = 0.0, cargo_fijo_por_cuota: float = 0.0,
            seguro_por_cuota: float = 0.0, iva_sobre_intereses_pct: float = 0.0) -> dict[str, Any]:
    """Préstamo en sistema francés: cuadro de marcha, TEA (3.3.1) y CFT (3.4) con sus componentes."""
    i = tna_pct * dias_periodo / (divisor * 100)  # tasa proporcional del período (3.3.1)
    cuota = cuota_frances(capital, i, cuotas)

    saldo, filas = capital, []
    for n in range(1, cuotas + 1):
        interes = saldo * i
        amortizacion = cuota - interes
        iva = interes * iva_sobre_intereses_pct / 100
        total = cuota + iva + cargo_fijo_por_cuota + seguro_por_cuota
        saldo -= amortizacion
        filas.append({"cuota": n, "interes": interes, "amortizacion": amortizacion, "iva": iva, "total": total,
                      "saldo": max(saldo, 0.0)})

    comision = capital * comision_otorgamiento_pct / 100
    desembolso_neto = capital - comision

    def cft_con(desembolso: float, extra_por_cuota: float, con_iva: bool) -> float | None:
        pagos = [cuota + (f["iva"] if con_iva else 0) + extra_por_cuota for f in filas]
        r = tir(desembolso, pagos)
        return None if r is None else anualizar(r, dias_periodo, divisor)

    cargos = cargo_fijo_por_cuota + seguro_por_cuota
    cft = cft_con(desembolso_neto, cargos, con_iva=True)
    return {
        "tasa_periodo_pct": i * 100,
        "tea_pct": tea_desde_nominal(tna_pct, dias_periodo, divisor),
        "cft_pct": cft,
        # Aporte de cada concepto al CFT, agregándolos de a uno (útil para explicar la diferencia TEA vs CFT)
        "cft_desglose_pct": {
            "solo_intereses (= TEA)": cft_con(capital, 0, con_iva=False),
            "+ comision_otorgamiento": cft_con(desembolso_neto, 0, con_iva=False),
            "+ cargos_y_seguro": cft_con(desembolso_neto, cargos, con_iva=False),
            "+ iva_sobre_intereses (= CFT)": cft,
        },
        "cuota_pura": cuota,
        "primera_cuota_total": filas[0]["total"],
        "ultima_cuota_total": filas[-1]["total"],
        "desembolso_neto": desembolso_neto,
        "comision_otorgamiento": comision,
        "total_intereses": sum(f["interes"] for f in filas),
        "total_iva": sum(f["iva"] for f in filas),
        "total_cargos_y_seguro": cargos * cuotas,
        "total_pagado": sum(f["total"] for f in filas),
        "cuadro": filas,
    }


# --- Herramientas -------------------------------------------------------------------------------------

def a_json(datos: dict[str, Any]) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=1)


def redondear(valor: Any, decimales: int = 2) -> Any:
    if isinstance(valor, float):
        return round(valor, decimales)
    if isinstance(valor, dict):
        return {k: redondear(v, decimales) for k, v in valor.items()}
    if isinstance(valor, list):
        return [redondear(v, decimales) for v in valor]
    return valor


def error(mensaje: str, **detalle: Any) -> str:
    return a_json({"estado": "datos_invalidos", "error": mensaje, **detalle,
                   "sugerencia": "Corregí los argumentos y reintentá. Si falta un dato del caso, no lo inventes: "
                                 "informá qué dato falta en tu respuesta."})


@tool
def convertir_tasa(tasa_nominal_anual_pct: float, dias_periodo: int = 30, divisor: Literal[365, 360] = 365,
                   modalidad: Literal["vencida", "adelantada"] = "vencida") -> str:
    """Convierte una tasa nominal anual (TNA) en tasa efectiva anual (TEA) con las fórmulas oficiales del
    BCRA (norma "Tasas de interés en las operaciones de crédito", punto 3.3).

    Usala cuando te pidan la TEA equivalente a una TNA, o para comparar tasas con distinta capitalización.
    No la uses para el CFT: el CFT incluye comisiones y cargos, para eso está `simular_prestamo`.

    Args:
        tasa_nominal_anual_pct: TNA pactada en tanto por ciento (ej. 60 para 60%).
        dias_periodo: días de cada subperíodo de pago de intereses (30 = mensual, 60 = bimestral). En una
            operación de pago único, los días totales de la operación.
        divisor: divisor fijo del punto 1.5: 365 en general; 360 solo en hipotecarios sobre vivienda y
            prendarios sobre automotores.
        modalidad: "vencida" (lo normal, punto 3.3.1) o "adelantada" (interés descontado al inicio, 3.3.2;
            en ese caso la tasa ingresada es la tasa de descuento anual).

    Returns:
        JSON con "estado" ("ok" o "datos_invalidos"), la TEA en % con dos decimales y la fórmula aplicada.
    """
    if not 0 <= tasa_nominal_anual_pct <= 1000:
        return error("La tasa debe estar entre 0 y 1000 (en %).", tasa_nominal_anual_pct=tasa_nominal_anual_pct)
    if dias_periodo <= 0:
        return error("dias_periodo debe ser positivo.", dias_periodo=dias_periodo)
    if modalidad == "adelantada" and tasa_nominal_anual_pct * dias_periodo / (divisor * 100) >= 1:
        return error("Con esa tasa de descuento el descuento supera el capital: revisá la tasa o el plazo.")
    tea = tea_desde_nominal(tasa_nominal_anual_pct, dias_periodo, divisor, modalidad)
    return a_json({
        "estado": "ok",
        "tna_pct": tasa_nominal_anual_pct,
        "tea_pct": round(tea, 2),
        "parametros": {"dias_periodo": dias_periodo, "divisor": divisor, "modalidad": modalidad},
        "formula": "t-tasint 3.3.1: i = [(1 + is·m/(df·100))^(df/m) − 1]·100" if modalidad == "vencida"
                   else "t-tasint 3.3.2: i = [(1/(1 − d·m/(df·100)))^(df/m) − 1]·100",
    })


@tool
def simular_prestamo(capital: float, tna_pct: float, cuotas: int, comision_otorgamiento_pct: float = 0.0,
                     cargo_fijo_por_cuota: float = 0.0, seguro_por_cuota: float = 0.0,
                     iva_sobre_intereses: bool = False, dias_periodo: int = 30,
                     divisor: Literal[365, 360] = 365) -> str:
    """Simula un préstamo en cuotas (sistema francés) y calcula la cuota, la TEA y el COSTO FINANCIERO
    TOTAL (CFT) según la normativa del BCRA (t-tasint, puntos 3.3 y 3.4).

    El CFT es la tasa efectiva anual que surge de los flujos reales del cliente: lo que recibe (capital
    menos comisiones cobradas al inicio) contra lo que paga en cada cuota (cuota pura + cargos + seguro
    + IVA sobre intereses si corresponde). Devuelve además el desglose de cuánto suma cada concepto.

    Usala cuando el caso tenga capital, tasa y cantidad de cuotas, y te pidan cuota, costo total o CFT.
    Antes de llamarla, fijate qué conceptos son computables en el CFT según la normativa (3.4.2): por
    ejemplo seguros exigidos para otorgar el crédito y cargos asociados SÍ; ciertos impuestos NO.

    Args:
        capital: monto prestado, en pesos.
        tna_pct: tasa nominal anual pactada, en % (ej. 60 para 60%).
        cuotas: cantidad de cuotas (1 a 600).
        comision_otorgamiento_pct: comisión cobrada al otorgar, en % del capital (se descuenta del monto
            recibido). 0 si no hay.
        cargo_fijo_por_cuota: cargos fijos en pesos que se suman a cada cuota (mantenimiento de cuenta
            asociada, envío de avisos, etc.). 0 si no hay.
        seguro_por_cuota: seguro exigido para el crédito, en pesos por cuota. 0 si no hay.
        iva_sobre_intereses: True si el cliente es consumidor final (IVA 21% sobre intereses, punto 3.4.2.7).
        dias_periodo: días entre cuotas (30 = mensual).
        divisor: 365 en general; 360 en hipotecarios sobre vivienda y prendarios sobre automotores.

    Returns:
        JSON con "estado" ("ok" o "datos_invalidos"), cuota pura, primera/última cuota total, totales,
        TEA y CFT en % (dos decimales), y "cft_desglose_pct" con el CFT acumulando cada concepto.
    """
    if capital <= 0:
        return error("El capital debe ser mayor a 0.", capital=capital)
    if not 1 <= cuotas <= MAX_CUOTAS:
        return error(f"La cantidad de cuotas debe estar entre 1 y {MAX_CUOTAS}.", cuotas=cuotas)
    if not 0 <= tna_pct <= 1000:
        return error("La TNA debe estar entre 0 y 1000 (en %).", tna_pct=tna_pct)
    if not 0 <= comision_otorgamiento_pct < 100:
        return error("La comisión de otorgamiento debe estar entre 0 y 100 (en % del capital).")
    if cargo_fijo_por_cuota < 0 or seguro_por_cuota < 0 or dias_periodo <= 0:
        return error("Cargos, seguro y dias_periodo no pueden ser negativos.")

    r = simular(capital, tna_pct, cuotas, dias_periodo, divisor, comision_otorgamiento_pct,
                cargo_fijo_por_cuota, seguro_por_cuota, IVA_GENERAL_PCT if iva_sobre_intereses else 0.0)
    if r["cft_pct"] is None:
        return error("No se pudo calcular el CFT con esos flujos (no existe una tasa que los iguale).")
    cuadro = r.pop("cuadro")
    return a_json({
        "estado": "ok",
        "datos": {"capital": capital, "tna_pct": tna_pct, "cuotas": cuotas,
                  "comision_otorgamiento_pct": comision_otorgamiento_pct,
                  "cargo_fijo_por_cuota": cargo_fijo_por_cuota, "seguro_por_cuota": seguro_por_cuota,
                  "iva_sobre_intereses": iva_sobre_intereses, "dias_periodo": dias_periodo, "divisor": divisor},
        **redondear(r),
        "cuadro_primeras_y_ultima": redondear([*cuadro[:2], cuadro[-1]]),
        "metodo": "Sistema francés con tasa proporcional (3.3.1). CFT = TIR de los flujos del cliente "
                  "anualizada con capitalización del período (3.4.1).",
    })


@tool
def calcular_cft_flujos(monto_recibido: float, pagos: list[float], dias_periodo: int = 30,
                        divisor: Literal[365, 360] = 365) -> str:
    """Calcula el CFT (tasa efectiva anual, %) a partir de flujos arbitrarios: lo que el cliente recibe
    hoy y la lista de pagos que hace en cada período. Sirve para casos que NO son sistema francés
    (cuotas desiguales, sistema alemán, pago único, cuotas con cargos variables).

    Si el caso es un préstamo en cuotas fijas, preferí `simular_prestamo`.

    Args:
        monto_recibido: lo que efectivamente recibe el cliente al inicio (capital menos gastos iniciales).
        pagos: pagos totales del cliente en cada período, en orden (incluyendo cargos, seguros, IVA).
        dias_periodo: días entre pagos (30 = mensual).
        divisor: 365 en general; 360 en hipotecarios sobre vivienda y prendarios sobre automotores.

    Returns:
        JSON con "estado" ("ok" o "datos_invalidos"), la tasa por período y el CFT en %.
    """
    if monto_recibido <= 0 or not pagos or any(p < 0 for p in pagos):
        return error("monto_recibido debe ser > 0 y pagos una lista no vacía de valores >= 0.")
    if sum(pagos) <= monto_recibido:
        # Matemáticamente hay una TIR (negativa), pero un "costo" negativo no es un CFT: casi seguro faltan pagos
        return error("Los pagos suman menos que el monto recibido: el costo sería negativo. ¿Faltan pagos o "
                     "están en otra unidad?", total_pagos=round(sum(pagos), 2), monto_recibido=monto_recibido)
    r = tir(monto_recibido, pagos)
    if r is None:
        return error("No existe una tasa que iguale los pagos con el monto recibido.",
                     total_pagos=round(sum(pagos), 2), monto_recibido=monto_recibido)
    return a_json({"estado": "ok", "tasa_periodo_pct": round(r * 100, 4),
                   "cft_pct": round(anualizar(r, dias_periodo, divisor), 2),
                   "total_pagado": round(sum(pagos), 2), "periodos": len(pagos)})


HERRAMIENTAS_ANALISTA = [convertir_tasa, simular_prestamo, calcular_cft_flujos]


# --- Prueba aislada: casos verificables a mano ---------------------------------------------------------

def main() -> None:
    def verificar(nombre: str, obtenido: float, esperado: float, tol: float = 0.01) -> None:
        print(f"  [{'OK' if abs(obtenido - esperado) <= tol else 'FALLA'}] {nombre}: {obtenido:.4f} (esperado {esperado:.4f})")

    print("1) Pago único: $1.000 a 30 días, TNA 36,5% -> interés 3% del período")
    r = json.loads(simular_prestamo.invoke({"capital": 1000, "tna_pct": 36.5, "cuotas": 1}))
    verificar("cuota", r["cuota_pura"], 1030)
    verificar("TEA = (1,03^(365/30) - 1)·100", r["tea_pct"], (1.03 ** (365 / 30) - 1) * 100)
    verificar("CFT sin gastos = TEA", r["cft_pct"], r["tea_pct"])

    print("2) Mismo caso con comisión 3%: recibe $970 y paga $1.030")
    r = json.loads(simular_prestamo.invoke({"capital": 1000, "tna_pct": 36.5, "cuotas": 1,
                                            "comision_otorgamiento_pct": 3}))
    verificar("CFT = ((1030/970)^(365/30) - 1)·100", r["cft_pct"], ((1030 / 970) ** (365 / 30) - 1) * 100)

    print("3) Caso de la demo: $1.000.000, 12 cuotas, TNA 60%, comisión 3%, seguro $2.000, consumidor final")
    r = json.loads(simular_prestamo.invoke({"capital": 1_000_000, "tna_pct": 60, "cuotas": 12,
                                            "comision_otorgamiento_pct": 3, "seguro_por_cuota": 2000,
                                            "iva_sobre_intereses": True}))
    i = 60 * 30 / 36500
    verificar("cuota pura = C·i/(1-(1+i)^-n)", r["cuota_pura"], 1_000_000 * i / (1 - (1 + i) ** -12))
    verificar("amortizaciones suman el capital",
              r["total_intereses"] + 1_000_000, r["cuota_pura"] * 12)
    print(f"  TEA {r['tea_pct']}% | CFT {r['cft_pct']}% | desglose {r['cft_desglose_pct']}")

    print("4) Conversión TNA 60% mensual y errores como datos")
    print("  ", json.loads(convertir_tasa.invoke({"tasa_nominal_anual_pct": 60}))["tea_pct"], "% TEA")
    print("  ", json.loads(simular_prestamo.invoke({"capital": 1000, "tna_pct": 60, "cuotas": 0}))["estado"])
    print("  ", json.loads(calcular_cft_flujos.invoke({"monto_recibido": 1000, "pagos": [100, 100]}))["estado"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
