# %% [markdown]
# # Notebook 08: Validación Cruzada (K-Fold)
#
# **Sección 15 - Tuning**: Cross-validation para evitar overfitting
#
# **Objetivo**: Implementar K-Fold Cross-Validation
#
# ## Conceptos clave:
# - Divide datos en K folds (subconjuntos)
# - Entrena K veces, usando diferente fold como validación
# - Promedia métricas para obtener estimación robusta
#
# ## Actividades:
# 1. Entender el concepto de K-Fold
# 2. Configurar CrossValidator en Spark ML
# 3. Combinar con ParamGrid para búsqueda de hiperparámetros
# 4. Analizar resultados

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

# %%
spark = SparkSession.builder \
    .appName("SECOP_CrossValidation") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %% [markdown]
# ## RETO 1: Entender K-Fold Cross-Validation
#
# **Pregunta conceptual**: Si usas K=5, responde:
#
# 1. ¿En cuántos subconjuntos se dividen los datos de train?
# 2. ¿Cuántos modelos se entrenan en total?
# 3. ¿Qué porcentaje de datos se usa para validación en cada iteración?
# 4. ¿Qué métrica se reporta al final?
#
# **Diagrama mental**:
# ```
# Fold 1: [VAL] [Train] [Train] [Train] [Train]
# Fold 2: [Train] [VAL] [Train] [Train] [Train]
# Fold 3: [Train] [Train] [VAL] [Train] [Train]
# Fold 4: [Train] [Train] [Train] [VAL] [Train]
# Fold 5: [Train] [Train] [Train] [Train] [VAL]
# ```
#
# **¿Por qué es mejor que un simple train/test split?**

# %%
# TODO: Escribe tus respuestas
# 1. Subconjuntos:
# 2. Modelos entrenados:
# 3. Porcentaje validación:
# 4. Métrica final:
# Ventaja sobre train/test simple:

# %% [markdown]
# ## RETO 2: Crear el Modelo Base y Evaluador
#
# **Objetivo**: Configurar LinearRegression y RegressionEvaluator.
#
# **Instrucciones**:
# 1. Crea un modelo de LinearRegression (sin hiperparámetros fijos)
# 2. Crea un evaluador con la métrica apropiada

# %%
# TODO: Crea el modelo base
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

# TODO: Crea el evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"  # TODO: ¿Usarías otra métrica?
)

# %% [markdown]
# ## RETO 3: Construir el ParamGrid
#
# **Objetivo**: Definir la grilla de hiperparámetros a explorar.
#
# **Instrucciones**:
# 1. Usa `ParamGridBuilder` para crear combinaciones
# 2. Incluye al menos regParam y elasticNetParam
# 3. Calcula cuántas combinaciones hay
#
# **Pregunta**: Si agregas 3 valores de regParam y 3 de elasticNetParam
# con K=5, ¿cuántos modelos se entrenan en total?
# Formula: combinaciones x K = ???

# %%
# TODO: Construye el grid de hiperparámetros
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [
        # TODO: Define valores de regularización a probar
    ]) \
    .addGrid(lr.elasticNetParam, [
        # TODO: Define valores de elasticNet a probar
    ]) \
    .build()

print(f"Combinaciones en el grid: {len(param_grid)}")
# TODO: Calcula total de modelos = combinaciones x K
# print(f"Total de modelos a entrenar: {len(param_grid) * K}")

# %% [markdown]
# ## RETO 4: Configurar CrossValidator
#
# **Objetivo**: Ensamblar el CrossValidator con modelo, grid y evaluador.
#
# **Parámetros clave**:
# - `estimator`: El modelo a entrenar
# - `estimatorParamMaps`: Las combinaciones de hiperparámetros
# - `evaluator`: Cómo evaluar cada modelo
# - `numFolds`: Valor de K (típicamente 3, 5 o 10)
# - `seed`: Para reproducibilidad
#
# **Pregunta**: ¿Qué valor de K elegirías?
# - K=3: Más rápido, menos robusto
# - K=5: Balance clásico
# - K=10: Más robusto, más lento
#
# **Consideración**: K grande en datasets grandes = MUY costoso

# %%
# TODO: Configura el CrossValidator
# crossval = CrossValidator(
#     estimator=lr,
#     estimatorParamMaps=param_grid,
#     evaluator=evaluator,
#     numFolds=???,  # TODO: Elige tu K
#     seed=42
# )

# print(f"Cross-Validation con K=??? folds")
# print(f"Total modelos a entrenar: {len(param_grid) * ???}")

# %% [markdown]
# ## RETO 5: Ejecutar Cross-Validation y Analizar
#
# **Objetivo**: Entrenar, obtener métricas y encontrar el mejor modelo.
#
# **Instrucciones**:
# 1. Ejecuta `crossval.fit(train)`
# 2. Obtén las métricas promedio con `cv_model.avgMetrics`
# 3. Identifica la mejor configuración
# 4. Evalúa el mejor modelo en test

# %%
# TODO: Ejecuta cross-validation
# print("Entrenando con Cross-Validation...")
# cv_model = crossval.fit(train)
# print("Cross-validation completada")

# %%
# TODO: Analiza las métricas promedio por configuración
# avg_metrics = cv_model.avgMetrics
# best_metric_idx = avg_metrics.index(min(avg_metrics))
#
# print("\n=== MÉTRICAS PROMEDIO POR CONFIGURACIÓN ===")
# for i, metric in enumerate(avg_metrics):
#     params = param_grid[i]
#     reg = params.get(lr.regParam)
#     elastic = params.get(lr.elasticNetParam)
#     marker = " <-- MEJOR" if i == best_metric_idx else ""
#     print(f"Config {i+1}: λ={reg:.2f}, α={elastic:.1f} -> RMSE={metric:,.2f}{marker}")

# %%
# TODO: Obtén y evalúa el mejor modelo
# best_model = cv_model.bestModel
#
# print("\n=== MEJOR MODELO ===")
# print(f"regParam: {best_model.getRegParam()}")
# print(f"elasticNetParam: {best_model.getElasticNetParam()}")
#
# predictions = best_model.transform(test)
# rmse_test = evaluator.evaluate(predictions)
# print(f"RMSE en Test: ${rmse_test:,.2f}")

# %% [markdown]
# ## RETO 6: Comparar CV vs Simple Split
#
# **Objetivo**: Demostrar la ventaja de cross-validation.
#
# **Instrucciones**:
# 1. Entrena un modelo con los mismos hiperparámetros pero SIN CV
# 2. Compara el RMSE de test con el modelo de CV
# 3. ¿Cuál es más confiable?
#
# **Pregunta**: ¿La métrica de CV es más cercana al rendimiento real? ¿Por qué?

# %%
# TODO: Entrena un modelo simple (sin CV) y compara
# lr_simple = LinearRegression(
#     featuresCol="features", labelCol="label",
#     maxIter=100, regParam=best_model.getRegParam(),
#     elasticNetParam=best_model.getElasticNetParam()
# )
# model_simple = lr_simple.fit(train)
# rmse_simple = evaluator.evaluate(model_simple.transform(test))
#
# print(f"RMSE con CV:     ${rmse_test:,.2f}")
# print(f"RMSE sin CV:     ${rmse_simple:,.2f}")
# print(f"Diferencia:      ${abs(rmse_test - rmse_simple):,.2f}")

# TODO: Responde:
# ¿Los resultados son similares? ¿Cuál método es más confiable?
# Respuesta:

# %% [markdown]
# ## RETO BONUS: Experimentar con diferentes K
#
# **Objetivo**: Observar el efecto del número de folds.
#
# **Instrucciones**:
# 1. Ejecuta CV con K=3, K=5, K=10
# 2. Compara la métrica promedio del mejor modelo en cada caso
# 3. Mide el tiempo de ejecución de cada uno
#
# **Pregunta**: ¿Más folds siempre es mejor?

# %%
# TODO: Experimenta con diferentes valores de K
# import time
#
# for k in [3, 5, 10]:
#     cv_temp = CrossValidator(
#         estimator=lr, estimatorParamMaps=param_grid,
#         evaluator=evaluator, numFolds=k, seed=42
#     )
#     start = time.time()
#     cv_temp_model = cv_temp.fit(train)
#     elapsed = time.time() - start
#     best_rmse = min(cv_temp_model.avgMetrics)
#     print(f"K={k:2d} | Mejor RMSE: ${best_rmse:,.2f} | Tiempo: {elapsed:.1f}s")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cuándo usarías K=3 vs K=10?**
#    Respuesta:
#
# 2. **¿Cross-validation reemplaza la necesidad de un test set?**
#    Respuesta:
#
# 3. **¿Qué pasa si tu dataset tiene solo 100 registros? ¿Qué K usarías?**
#    Respuesta:
#
# 4. **¿Es posible hacer CV con time series? ¿Qué cambiaría?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
# Guardar mejor modelo
# model_path = "/opt/spark-data/processed/cv_best_model"
# best_model.save(model_path)
# print(f"Modelo guardado en: {model_path}")

# %%
print("\n" + "="*60)
print("RESUMEN VALIDACIÓN CRUZADA")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Entendido el concepto de K-Fold")
print("  [ ] Configurado ParamGrid con hiperparámetros")
print("  [ ] Ejecutado CrossValidator")
print("  [ ] Identificado el mejor modelo")
print("  [ ] Comparado con entrenamiento simple")
print("="*60)

# %%
spark.stop()
