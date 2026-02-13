# %% [markdown]
# # Notebook 07: Regularización L1, L2 y ElasticNet
#
# **Sección 14**: Prevención de overfitting con regularización
#
# **Objetivo**: Comparar Ridge (L2), Lasso (L1) y ElasticNet
#
# ## Conceptos clave:
# - **Ridge (L2)**: regParam > 0, elasticNetParam = 0
#   - Penaliza coeficientes grandes, NO los elimina
# - **Lasso (L1)**: regParam > 0, elasticNetParam = 1
#   - Puede eliminar features (coeficientes = 0)
# - **ElasticNet**: regParam > 0, elasticNetParam ∈ (0, 1)
#   - Combinación de L1 y L2
#
# ## Actividades:
# 1. Entrenar modelos con diferentes regularizaciones
# 2. Comparar resultados
# 3. Identificar el mejor modelo

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import pandas as pd

# %%
spark = SparkSession.builder \
    .appName("SECOP_Regularizacion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %% [markdown]
# ## RETO 1: Entender la Regularización
#
# **Pregunta conceptual**: ¿Por qué necesitamos regularización?
#
# **Escenario**: Tu modelo de regresión lineal tiene:
# - R² train = 0.95
# - R² test = 0.45
#
# **Opciones**:
# - A) El modelo está underfitting
# - B) El modelo está overfitting
# - C) El modelo es perfecto
# - D) Necesitas más features
#
# **¿Cómo ayuda la regularización en este caso?**
#
# **Responde antes de continuar**

# %%
# TODO: Escribe tu respuesta
# El escenario indica:
# La regularización ayuda porque:

# %% [markdown]
# ## RETO 2: Configurar el Evaluador
#
# **Objetivo**: Crear un evaluador para comparar modelos.
#
# **Pregunta**: ¿Qué métrica usarías para comparar modelos de regresión?
# - RMSE: Penaliza errores grandes
# - MAE: Trata todos los errores igual
# - R²: Proporción de varianza explicada

# %%
# TODO: Configura el evaluador
# Pista: RegressionEvaluator con metricName apropiado

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"  # TODO: ¿Cambiarías esta métrica? ¿Por qué?
)

# %% [markdown]
# ## RETO 3: Experimento de Regularización
#
# **Objetivo**: Entrenar múltiples modelos variando regParam y elasticNetParam.
#
# **Instrucciones**:
# 1. Define los valores de `regParam` (lambda) a probar
# 2. Define los valores de `elasticNetParam` (alpha) a probar
# 3. Entrena un modelo por cada combinación
# 4. Registra RMSE de train y test para cada uno
#
# **Parámetros sugeridos**:
# - regParam: [0.0, 0.01, 0.1, 1.0, 10.0]
# - elasticNetParam: [0.0 (Ridge), 0.5 (ElasticNet), 1.0 (Lasso)]
#
# **Pregunta**: ¿Cuántos modelos entrenarás en total? (combinaciones)

# %%
# TODO: Define los valores a probar
reg_params = [
    # TODO: Completa con tus valores de lambda
]

elastic_params = [
    # TODO: Completa con tus valores de alpha
    # Recuerda: 0.0=Ridge, 0.5=ElasticNet, 1.0=Lasso
]

print(f"Combinaciones totales: {len(reg_params) * len(elastic_params)}")

# %%
# TODO: Implementa el bucle de experimentación
# Pista: Itera sobre reg_params y elastic_params
# Para cada combinación:
#   1. Crea LinearRegression con esos parámetros
#   2. Entrena con train
#   3. Evalúa en test
#   4. Guarda resultados en una lista

resultados = []

# for reg in reg_params:
#     for elastic in elastic_params:
#         lr = LinearRegression(
#             featuresCol="features",
#             labelCol="label",
#             maxIter=100,
#             regParam=reg,
#             elasticNetParam=elastic
#         )
#
#         model = lr.fit(train)
#         predictions = model.transform(test)
#         rmse = evaluator.evaluate(predictions)
#
#         # Determinar tipo de regularización
#         if reg == 0.0:
#             reg_type = "Sin regularización"
#         elif elastic == 0.0:
#             reg_type = "Ridge (L2)"
#         elif elastic == 1.0:
#             reg_type = "Lasso (L1)"
#         else:
#             reg_type = "ElasticNet"
#
#         resultados.append({
#             "regParam": reg,
#             "elasticNetParam": elastic,
#             "tipo": reg_type,
#             "rmse_test": rmse,
#             "rmse_train": model.summary.rootMeanSquaredError,
#             "r2_train": model.summary.r2
#         })
#
#         print(f"{reg_type:25s} | λ={reg:5.2f} | α={elastic:.1f} | RMSE Test: ${rmse:,.2f}")

# %% [markdown]
# ## RETO 4: Analizar Resultados
#
# **Objetivo**: Comparar todos los modelos y encontrar el mejor.
#
# **Instrucciones**:
# 1. Convierte los resultados a un DataFrame de pandas
# 2. Ordena por RMSE test
# 3. Identifica el mejor modelo
# 4. Compara RMSE train vs test para detectar overfitting
#
# **Pregunta**: ¿El mejor modelo es siempre el que tiene menor RMSE en test?
# ¿Qué otros factores considerarías?

# %%
# TODO: Convierte resultados a DataFrame y analiza
# df_resultados = pd.DataFrame(resultados)
# print(df_resultados.to_string(index=False))

# TODO: Encuentra el mejor modelo
# mejor_modelo = df_resultados.loc[df_resultados['rmse_test'].idxmin()]
# print(f"\nMejor modelo: {mejor_modelo['tipo']}")
# print(f"  regParam: {mejor_modelo['regParam']}")
# print(f"  RMSE Test: ${mejor_modelo['rmse_test']:,.2f}")

# %% [markdown]
# ## RETO 5: Comparar Overfitting
#
# **Objetivo**: Analizar la brecha entre train y test para cada tipo de regularización.
#
# **Instrucciones**:
# 1. Calcula la diferencia RMSE_test - RMSE_train para cada modelo
# 2. ¿Qué tipo de regularización reduce más el overfitting?
# 3. ¿Hay un trade-off entre overfitting y rendimiento general?
#
# **Pregunta de análisis**:
# - Si regParam=0.0 tiene RMSE_train muy bajo pero RMSE_test alto → ¿overfitting?
# - Si regParam=10.0 tiene RMSE_train y RMSE_test ambos altos → ¿underfitting?

# %%
# TODO: Calcula y analiza la brecha train-test
# for _, row in df_resultados.iterrows():
#     gap = row['rmse_test'] - row['rmse_train']
#     print(f"{row['tipo']:25s} | Gap: ${gap:,.2f} | Overfitting: {'Sí' if gap > threshold else 'No'}")

# TODO: Responde:
# ¿Qué regularización reduce más el overfitting?
# Respuesta:

# %% [markdown]
# ## RETO 6: Entrenar Modelo Final
#
# **Objetivo**: Entrenar el modelo con los mejores hiperparámetros.
#
# **Instrucciones**:
# 1. Usa los hiperparámetros del mejor modelo encontrado
# 2. Entrena con todos los datos de train
# 3. Evalúa en test
# 4. Guarda el modelo

# %%
# TODO: Entrena el modelo final con los mejores hiperparámetros
# lr_final = LinearRegression(
#     featuresCol="features",
#     labelCol="label",
#     maxIter=100,
#     regParam=???,           # TODO: Usa el mejor regParam
#     elasticNetParam=???     # TODO: Usa el mejor elasticNetParam
# )
#
# modelo_final = lr_final.fit(train)

# %%
# TODO: Guarda el mejor modelo
# model_path = "/opt/spark-data/processed/regularized_model"
# modelo_final.save(model_path)
# print(f"Mejor modelo guardado en: {model_path}")

# %% [markdown]
# ## RETO BONUS: Efecto de Lambda en los Coeficientes
#
# **Objetivo**: Visualizar cómo lambda afecta los coeficientes del modelo.
#
# **Instrucciones**:
# 1. Para cada valor de regParam, entrena un modelo Lasso (L1)
# 2. Extrae los coeficientes
# 3. Cuenta cuántos coeficientes son exactamente 0
# 4. ¿A mayor lambda, más coeficientes eliminados?
#
# **Pregunta**: ¿Por qué Lasso puede poner coeficientes en 0 pero Ridge no?

# %%
# TODO: Implementa el análisis de coeficientes
# import numpy as np
#
# for reg in [0.01, 0.1, 1.0, 10.0]:
#     lr_lasso = LinearRegression(
#         featuresCol="features", labelCol="label",
#         maxIter=100, regParam=reg, elasticNetParam=1.0
#     )
#     model_lasso = lr_lasso.fit(train)
#     coefs = np.array(model_lasso.coefficients)
#     zeros = np.sum(np.abs(coefs) < 1e-6)
#     print(f"λ={reg:5.2f} | Coeficientes=0: {zeros}/{len(coefs)} | RMSE: ${evaluator.evaluate(model_lasso.transform(test)):,.2f}")

# TODO: Responde:
# ¿Por qué Lasso elimina features y Ridge no?
# Respuesta:

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cuándo usarías Ridge vs Lasso vs ElasticNet?**
#    Respuesta:
#
# 2. **¿Qué pasa si regParam es demasiado grande?**
#    Respuesta:
#
# 3. **¿Es posible que el modelo sin regularización sea el mejor?**
#    Respuesta:
#
# 4. **¿Cómo elegirías el valor óptimo de regParam en producción?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
# Guardar resultados del experimento
# import json
# with open("/opt/spark-data/processed/regularizacion_resultados.json", "w") as f:
#     json.dump(resultados, f, indent=2)
# print("Resultados guardados")

# %%
print("\n" + "="*60)
print("RESUMEN REGULARIZACIÓN")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Entendido diferencia entre L1, L2 y ElasticNet")
print("  [ ] Experimentado con múltiples combinaciones")
print("  [ ] Identificado el mejor modelo")
print("  [ ] Analizado overfitting vs underfitting")
print("  [ ] Guardado modelo final")
print("="*60)

# %%
spark.stop()
