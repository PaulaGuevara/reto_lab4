# %% [markdown]
# # Notebook 11: Model Registry con MLflow
#
# **Sección 16 - MLOps**: Versionamiento y gestión del ciclo de vida
#
# **Objetivo**: Registrar modelos, crear versiones y promover a producción
#
# ## Conceptos clave:
# - **Model Registry**: Catálogo centralizado de modelos
# - **Versioning**: Cada modelo puede tener múltiples versiones (v1, v2, etc.)
# - **Stages**: Ciclo de vida: None -> Staging -> Production -> Archived
# - **MlflowClient**: API programática para gestionar el registry
#
# ## Actividades:
# 1. Registrar modelo en MLflow Model Registry
# 2. Crear versiones (v1, v2, etc.)
# 3. Transicionar entre stages: None -> Staging -> Production
# 4. Cargar modelo desde Registry

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# %%
spark = SparkSession.builder \
    .appName("SECOP_ModelRegistry") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %% [markdown]
# ## RETO 1: Configurar MLflow y el Registry
#
# **Objetivo**: Conectar al tracking server y preparar el Model Registry.
#
# **Instrucciones**:
# 1. Configura la URI del tracking server
# 2. Crea un `MlflowClient` para interactuar con el registry
# 3. Define un nombre descriptivo para tu modelo
#
# **Pregunta**: ¿Qué diferencia hay entre el Tracking Server y el Model Registry?
# - Tracking: Registra experimentos individuales (runs)
# - Registry: Gestiona modelos versionados y su ciclo de vida

# %%
# TODO: Configura MLflow y el cliente del registry
# mlflow.set_tracking_uri("http://mlflow:5000")
# client = MlflowClient()

# TODO: Define el nombre del modelo
# model_name = "secop_prediccion_contratos"

# print(f"MLflow URI: {mlflow.get_tracking_uri()}")
# print(f"Modelo: {model_name}")

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# %% [markdown]
# ## RETO 2: Entrenar y Registrar Modelo v1 (Baseline)
#
# **Objetivo**: Entrenar un modelo baseline y registrarlo como versión 1.
#
# **Instrucciones**:
# 1. Configura el experimento
# 2. Entrena un modelo SIN regularización
# 3. Evalúa y registra métricas
# 4. Registra el modelo en el registry con `registered_model_name`
#
# **Concepto clave**: Al usar `registered_model_name` en `log_model()`,
# MLflow automáticamente crea el modelo en el registry si no existe,
# o agrega una nueva versión si ya existe.

# %%
# TODO: Entrena y registra modelo v1
# mlflow.set_experiment("/SECOP_Model_Registry")
#
# with mlflow.start_run(run_name="model_v1_baseline") as run:
#     # TODO: Entrena modelo baseline (sin regularización)
#     # lr = LinearRegression(featuresCol="features", labelCol="label",
#     #                       regParam=0.0, maxIter=100)
#     # model_v1 = lr.fit(train)
#
#     # TODO: Evalúa
#     # predictions = model_v1.transform(test)
#     # rmse_v1 = evaluator.evaluate(predictions)
#
#     # TODO: Log de parámetros y métricas
#     # mlflow.log_param("version", "1.0")
#     # mlflow.log_param("model_type", "baseline")
#     # mlflow.log_metric("rmse", rmse_v1)
#
#     # TODO: Registra el modelo en el Registry
#     # mlflow.spark.log_model(
#     #     spark_model=model_v1,
#     #     artifact_path="model",
#     #     registered_model_name=model_name  # <-- Esto registra en el Registry
#     # )
#
#     # run_id_v1 = run.info.run_id
#     # print(f"Modelo v1 registrado - Run ID: {run_id_v1}, RMSE: ${rmse_v1:,.2f}")

# %% [markdown]
# ## RETO 3: Entrenar y Registrar Modelo v2 (Mejorado)
#
# **Objetivo**: Entrenar un modelo mejorado y registrarlo como versión 2.
#
# **Instrucciones**:
# 1. Entrena un modelo CON regularización (usa los mejores hiperparámetros del notebook 09)
# 2. Evalúa y compara con v1
# 3. Registra como nueva versión del mismo modelo
#
# **Pregunta**: ¿Por qué versionar modelos en lugar de sobrescribir?

# %%
# TODO: Entrena y registra modelo v2 (con regularización)
# with mlflow.start_run(run_name="model_v2_regularized") as run:
#     # TODO: Entrena modelo con regularización
#     # lr = LinearRegression(featuresCol="features", labelCol="label",
#     #                       regParam=0.1, elasticNetParam=0.5, maxIter=100)
#     # model_v2 = lr.fit(train)
#
#     # TODO: Evalúa
#     # rmse_v2 = evaluator.evaluate(model_v2.transform(test))
#
#     # TODO: Log y registro
#     # mlflow.log_param("version", "2.0")
#     # mlflow.log_param("model_type", "regularized")
#     # mlflow.log_metric("rmse", rmse_v2)
#     # mlflow.spark.log_model(spark_model=model_v2, artifact_path="model",
#     #                        registered_model_name=model_name)
#
#     # print(f"Modelo v2 registrado - RMSE: ${rmse_v2:,.2f}")

# TODO: Compara v1 vs v2
# print(f"\nComparación:")
# print(f"  v1 RMSE: ${rmse_v1:,.2f}")
# print(f"  v2 RMSE: ${rmse_v2:,.2f}")
# print(f"  Mejor: {'v2' if rmse_v2 < rmse_v1 else 'v1'}")

# %% [markdown]
# ## RETO 4: Gestionar Versiones y Stages
#
# **Objetivo**: Transicionar modelos entre stages del ciclo de vida.
#
# **Ciclo de vida**:
# ```
# None -> Staging -> Production -> Archived
# ```
#
# **Instrucciones**:
# 1. Lista las versiones registradas del modelo
# 2. Promueve la mejor versión a "Staging"
# 3. Si pasa la validación, promuévela a "Production"
# 4. Archiva la versión anterior
#
# **Pregunta**: ¿Por qué pasar por Staging antes de Production?

# %%
# TODO: Lista las versiones del modelo
# model_versions = client.search_model_versions(f"name='{model_name}'")
# print(f"Versiones del modelo '{model_name}':")
# for mv in model_versions:
#     print(f"  - Versión {mv.version}: Stage={mv.current_stage}, Run={mv.run_id[:8]}")

# %%
# TODO: Transiciona la versión 1 a Staging
# client.transition_model_version_stage(
#     name=model_name,
#     version=1,
#     stage="Staging"
# )
# print("v1 -> Staging")

# TODO: Si v2 es mejor, promuévela a Production
# if rmse_v2 < rmse_v1:
#     client.transition_model_version_stage(
#         name=model_name, version=2, stage="Production"
#     )
#     print("v2 -> Production (mejor modelo)")
#
#     # Archiva v1
#     client.transition_model_version_stage(
#         name=model_name, version=1, stage="Archived"
#     )
#     print("v1 -> Archived")

# TODO: Responde:
# ¿Por qué pasar por Staging antes de Production?
# Respuesta:

# %% [markdown]
# ## RETO 5: Agregar Metadata al Modelo
#
# **Objetivo**: Documentar el modelo con descripciones y etiquetas.
#
# **Instrucciones**:
# 1. Agrega una descripción a la versión en producción
# 2. Incluye información útil: RMSE, fecha, autor, dataset usado
#
# **Pregunta**: ¿Qué información mínima debería tener cada versión de modelo?

# %%
# TODO: Agrega metadata al modelo en producción
# best_version = 2 if rmse_v2 < rmse_v1 else 1
#
# client.update_model_version(
#     name=model_name,
#     version=best_version,
#     description=f"Modelo para predicción de contratos SECOP. RMSE: ${min(rmse_v1, rmse_v2):,.2f}"
# )
# print(f"Metadata agregada a versión {best_version}")

# %% [markdown]
# ## RETO 6: Cargar Modelo desde Registry
#
# **Objetivo**: Cargar el modelo en producción para hacer predicciones.
#
# **Concepto**: En producción, cargamos modelos por su nombre y stage,
# NO por ruta de archivo. Esto permite:
# - Cambiar la versión en producción sin modificar código
# - Rollback instantáneo si algo falla
#
# **Instrucciones**:
# 1. Carga el modelo desde el Registry usando `models:/{name}/{stage}`
# 2. Verifica que funciona haciendo predicciones en test
# 3. Compara el RMSE con el esperado

# %%
# TODO: Carga el modelo desde el Registry
# model_uri = f"models:/{model_name}/Production"
# loaded_model = mlflow.spark.load_model(model_uri)
#
# print(f"Modelo cargado desde: {model_uri}")
# print(f"Tipo: {type(loaded_model)}")

# TODO: Verifica que funciona
# test_predictions = loaded_model.transform(test)
# test_rmse = evaluator.evaluate(test_predictions)
# print(f"RMSE verificación: ${test_rmse:,.2f}")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cómo harías rollback si el modelo en Production falla?**
#    Respuesta:
#
# 2. **¿Qué criterios usarías para promover un modelo de Staging a Production?**
#    Respuesta:
#
# 3. **¿Cómo implementarías A/B testing con el Model Registry?**
#    Respuesta:
#
# 4. **¿Quién debería tener permisos para promover modelos a Production?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
print("\n" + "="*60)
print("RESUMEN MODEL REGISTRY")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Registrado modelo v1 (baseline)")
print("  [ ] Registrado modelo v2 (mejorado)")
print("  [ ] Transicionado versiones entre stages")
print("  [ ] Agregado metadata al modelo")
print("  [ ] Cargado modelo desde Registry")
print("  [ ] Accede a Model Registry: http://localhost:5000/#/models")
print("="*60)

# %%
spark.stop()
