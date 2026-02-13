# %% [markdown]
# # Notebook 09: Optimización de Hiperparámetros
#
# **Sección 15**: Grid Search y Train-Validation Split
#
# **Objetivo**: Encontrar la mejor combinación de hiperparámetros
#
# ## Estrategias:
# - **Grid Search + CV**: Búsqueda exhaustiva con cross-validation
# - **Train-Validation Split**: Alternativa más rápida (un solo split)
#
# ## Actividades:
# 1. Implementar Grid Search exhaustivo
# 2. Implementar Train-Validation Split
# 3. Comparar ambas estrategias
# 4. Seleccionar el mejor modelo global

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col
import time

# %%
spark = SparkSession.builder \
    .appName("SECOP_HyperparameterTuning") \
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

# %%
# Modelo base y evaluador
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# %% [markdown]
# ## RETO 1: Diseñar el Grid de Hiperparámetros
#
# **Objetivo**: Crear un grid amplio pero inteligente.
#
# **Parámetros a explorar**:
# - `regParam`: Fuerza de regularización [0.01, 0.1, 1.0]
# - `elasticNetParam`: Tipo de regularización [0.0, 0.5, 1.0]
# - `maxIter`: Iteraciones máximas [50, 100, 200]
#
# **Pregunta de diseño**: ¿Por qué usamos escala logarítmica para regParam
# (0.01, 0.1, 1.0) en lugar de lineal (0.33, 0.66, 1.0)?
#
# **Calcula**: ¿Cuántas combinaciones genera tu grid?

# %%
# TODO: Construye el grid de hiperparámetros
grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [
        # TODO: Define valores de regularización
    ]) \
    .addGrid(lr.elasticNetParam, [
        # TODO: Define valores de elasticNet
    ]) \
    .addGrid(lr.maxIter, [
        # TODO: Define valores de iteraciones máximas
    ]) \
    .build()

print(f"Combinaciones totales: {len(grid)}")

# TODO: Responde:
# ¿Cuántas combinaciones hay?
# ¿Cuántos modelos se entrenan con K=3?
# Respuesta:

# %% [markdown]
# ## RETO 2: Grid Search con Cross-Validation
#
# **Objetivo**: Ejecutar búsqueda exhaustiva con K-Fold CV.
#
# **Instrucciones**:
# 1. Configura CrossValidator con tu grid
# 2. Usa K=3 (balance entre robustez y velocidad)
# 3. Ejecuta el entrenamiento
# 4. Registra el tiempo de ejecución
# 5. Obtén el mejor modelo y su RMSE en test
#
# **Pregunta**: ¿Por qué K=3 y no K=5 para este experimento?

# %%
# TODO: Configura y ejecuta Grid Search + CV
# cv_grid = CrossValidator(
#     estimator=lr,
#     estimatorParamMaps=grid,
#     evaluator=evaluator,
#     numFolds=3,
#     seed=42
# )
#
# print("Entrenando Grid Search + CV...")
# start_time = time.time()
# cv_grid_model = cv_grid.fit(train)
# grid_time = time.time() - start_time
#
# print(f"Completado en {grid_time:.2f} segundos")

# %%
# TODO: Obtén y evalúa el mejor modelo
# best_grid_model = cv_grid_model.bestModel
# predictions_grid = best_grid_model.transform(test)
# rmse_grid = evaluator.evaluate(predictions_grid)
#
# print("\n=== MEJOR MODELO (Grid Search + CV) ===")
# print(f"regParam: {best_grid_model.getRegParam()}")
# print(f"elasticNetParam: {best_grid_model.getElasticNetParam()}")
# print(f"maxIter: {best_grid_model.getMaxIter()}")
# print(f"RMSE Test: ${rmse_grid:,.2f}")

# %% [markdown]
# ## RETO 3: Train-Validation Split
#
# **Objetivo**: Implementar una alternativa más rápida al CV completo.
#
# **Concepto**: TrainValidationSplit divide los datos de entrenamiento en
# un único split train/validation (ej: 80/20), en lugar de K folds.
#
# **Ventaja**: Solo entrena cada modelo 1 vez (vs K veces en CV)
# **Desventaja**: Menos robusto (depende de un solo split)
#
# **Instrucciones**:
# 1. Usa TrainValidationSplit con el mismo grid
# 2. Configura trainRatio=0.8
# 3. Ejecuta y registra el tiempo
# 4. Compara con Grid Search + CV

# %%
# TODO: Configura y ejecuta Train-Validation Split
# tvs = TrainValidationSplit(
#     estimator=lr,
#     estimatorParamMaps=grid,
#     evaluator=evaluator,
#     trainRatio=???,  # TODO: ¿Qué ratio usarías?
#     seed=42
# )
#
# print("Entrenando con Train-Validation Split...")
# start_time = time.time()
# tvs_model = tvs.fit(train)
# tvs_time = time.time() - start_time
#
# print(f"Completado en {tvs_time:.2f} segundos")

# %%
# TODO: Obtén y evalúa el mejor modelo
# best_tvs_model = tvs_model.bestModel
# predictions_tvs = best_tvs_model.transform(test)
# rmse_tvs = evaluator.evaluate(predictions_tvs)
#
# print("\n=== MEJOR MODELO (Train-Validation Split) ===")
# print(f"regParam: {best_tvs_model.getRegParam()}")
# print(f"elasticNetParam: {best_tvs_model.getElasticNetParam()}")
# print(f"maxIter: {best_tvs_model.getMaxIter()}")
# print(f"RMSE Test: ${rmse_tvs:,.2f}")

# %% [markdown]
# ## RETO 4: Comparar Estrategias
#
# **Objetivo**: Analizar las diferencias entre Grid Search + CV y Train-Validation Split.
#
# **Instrucciones**:
# 1. Compara RMSE de ambas estrategias
# 2. Compara tiempos de ejecución
# 3. ¿Eligieron los mismos hiperparámetros?
# 4. ¿Cuándo usarías cada estrategia?

# %%
# TODO: Compara ambas estrategias
# print("\n" + "="*60)
# print("COMPARACIÓN DE ESTRATEGIAS")
# print("="*60)
# print(f"Grid Search + CV:")
# print(f"  - Tiempo: {grid_time:.2f}s")
# print(f"  - RMSE Test: ${rmse_grid:,.2f}")
# print(f"  - Hiperparámetros: λ={best_grid_model.getRegParam()}, α={best_grid_model.getElasticNetParam()}")
# print(f"\nTrain-Validation Split:")
# print(f"  - Tiempo: {tvs_time:.2f}s")
# print(f"  - RMSE Test: ${rmse_tvs:,.2f}")
# print(f"  - Hiperparámetros: λ={best_tvs_model.getRegParam()}, α={best_tvs_model.getElasticNetParam()}")

# TODO: Responde:
# ¿Qué estrategia tiene mejor rendimiento?
# ¿Cuál es más rápida?
# ¿Eligieron los mismos hiperparámetros?
# Respuesta:

# %% [markdown]
# ## RETO 5: Seleccionar y Guardar Modelo Final
#
# **Objetivo**: Guardar el mejor modelo global y sus hiperparámetros.
#
# **Instrucciones**:
# 1. Selecciona el mejor modelo entre ambas estrategias
# 2. Guarda el modelo en disco
# 3. Guarda los hiperparámetros en un JSON

# %%
# TODO: Selecciona y guarda el mejor modelo
# mejor_modelo = best_grid_model if rmse_grid < rmse_tvs else best_tvs_model
# model_path = "/opt/spark-data/processed/tuned_model"
# mejor_modelo.save(model_path)
# print(f"Mejor modelo guardado en: {model_path}")

# %%
# TODO: Guarda los hiperparámetros óptimos
# import json
#
# hiperparametros_optimos = {
#     "regParam": float(mejor_modelo.getRegParam()),
#     "elasticNetParam": float(mejor_modelo.getElasticNetParam()),
#     "maxIter": int(mejor_modelo.getMaxIter()),
#     "rmse_test": float(rmse_grid if rmse_grid < rmse_tvs else rmse_tvs),
#     "estrategia": "Grid Search + CV" if rmse_grid < rmse_tvs else "Train-Validation Split"
# }
#
# with open("/opt/spark-data/processed/hiperparametros_optimos.json", "w") as f:
#     json.dump(hiperparametros_optimos, f, indent=2)
#
# print("Hiperparámetros óptimos guardados")

# %% [markdown]
# ## RETO BONUS: Grid Más Fino
#
# **Objetivo**: Refinar la búsqueda alrededor de los mejores hiperparámetros.
#
# **Concepto**: Una vez identificada la mejor zona, crea un grid más fino
# alrededor de esos valores.
#
# **Ejemplo**: Si el mejor regParam fue 0.1, prueba [0.05, 0.08, 0.1, 0.12, 0.15]
#
# **Instrucciones**:
# 1. Toma los mejores hiperparámetros encontrados
# 2. Crea un grid fino alrededor de esos valores
# 3. Ejecuta CV con el grid fino
# 4. ¿Mejora el RMSE?

# %%
# TODO: Implementa el refinamiento del grid
# Pista: Crea un nuevo ParamGridBuilder con valores más cercanos
# al mejor regParam y elasticNetParam encontrado

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cuándo usarías Grid Search vs Random Search?**
#    Respuesta:
#
# 2. **¿Por qué Train-Validation Split es más rápido que CV?**
#    Respuesta:
#
# 3. **¿Qué pasa si el grid es demasiado grande?**
#    Respuesta:
#
# 4. **¿Cómo implementarías Random Search en Spark ML?**
#    (Pista: Spark ML no tiene Random Search nativo)
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
print("\n" + "="*60)
print("RESUMEN OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Diseñado grid de hiperparámetros")
print("  [ ] Ejecutado Grid Search + CV")
print("  [ ] Ejecutado Train-Validation Split")
print("  [ ] Comparado ambas estrategias")
print("  [ ] Guardado mejor modelo y hiperparámetros")
print("="*60)

# %%
spark.stop()
