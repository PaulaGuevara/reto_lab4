# ============================================
# Estadísticas descriptivas robustas – SECOP II
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# 1. Función: resumen robusto para numéricas
# --------------------------------------------
def resumen_robusto_numericas(df, columnas):
    resumen = []

    for col in columnas:
        serie = pd.to_numeric(df[col], errors='coerce')

        resumen.append({
            'variable': col,
            'n': serie.notna().sum(),
            'missing_%': round(serie.isna().mean() * 100, 2),
            'mediana': serie.median(),
            'p10': serie.quantile(0.10),
            'p90': serie.quantile(0.90),
            'iqr': serie.quantile(0.75) - serie.quantile(0.25),
            'ceros_%': round((serie == 0).mean() * 100, 2)
        })

    return pd.DataFrame(resumen)


# --------------------------------------------
# 2. Función: resumen corto para categóricas
# --------------------------------------------
def resumen_categoricas(df, columnas, top=5):
    resumen = {}

    for col in columnas:
        resumen[col] = (
            df[col]
            .value_counts(normalize=True, dropna=False)
            .head(top)
            .mul(100)
            .round(2)
        )

    return resumen


# --------------------------------------------
# 3. Variables de interés (SECOP II)
# --------------------------------------------

vars_numericas = [
    'valor_del_contrato',
    'valor_pagado',
    'valor_pendiente_de_pago',
    'valor_facturado',
    'valor_amortizado',
    'saldo_cdp',
    'saldo_vigencia',
    'dias_adicionados'
]

vars_categoricas = [
    'tipo_de_contrato',
    'modalidad_de_contratacion',
    'estado_contrato',
    'sector',
    'orden',
    'es_pyme',
    'entidad_centralizada'
]


# --------------------------------------------
# 4. Ejecución de resúmenes
# --------------------------------------------

resumen_numerico = resumen_robusto_numericas(df, vars_numericas)
resumen_cat = resumen_categoricas(df, vars_categoricas)


# --------------------------------------------
# 5. Análisis temporal básico
# --------------------------------------------

df['fecha_de_firma'] = pd.to_datetime(df['fecha_de_firma'], errors='coerce')
df['fecha_de_inicio_del_contrato'] = pd.to_datetime(
    df['fecha_de_inicio_del_contrato'], errors='coerce'
)
df['fecha_de_fin_del_contrato'] = pd.to_datetime(
    df['fecha_de_fin_del_contrato'], errors='coerce'
)

rango_fechas = {
    'inicio_contratos': df['fecha_de_firma'].min(),
    'fin_contratos': df['fecha_de_firma'].max()
}

df['duracion_dias'] = (
    df['fecha_de_fin_del_contrato'] - df['fecha_de_inicio_del_contrato']
).dt.days

duracion_mediana = df['duracion_dias'].median()


# --------------------------------------------
# 6. Outputs finales (listos para reporte)
# --------------------------------------------

print("\n=== RESUMEN ROBUSTO NUMÉRICO ===")
print(resumen_numerico)

print("\n=== RESUMEN CATEGÓRICO (Top 5 %) ===")
for k, v in resumen_cat.items():
    print(f"\n{k}")
    print(v)

print("\n=== RANGO TEMPORAL DE LOS CONTRATOS ===")
print(rango_fechas)

print("\n=== DURACIÓN MEDIANA DEL CONTRATO (días) ===")
print(duracion_mediana)
