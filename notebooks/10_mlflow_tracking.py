# %% [markdown]
# # Notebook 10: MLflow Tracking
#
# **Sección 16 - MLOps**: Registro de experimentos con MLflow
#
# **Objetivo**: Rastrear experimentos, métricas y modelos con MLflow
#
# ## Conceptos clave:
# - **Experiment**: Agrupación lógica de runs (un proyecto)
# - **Run**: Una ejecución individual (un modelo entrenado)
# - **Parameters**: Hiperparámetros registrados (regParam, maxIter, etc.)
# - **Metrics**: Métricas de rendimiento (RMSE, R², etc.)
# - **Artifacts**: Archivos guardados (modelos, gráficos, etc.)
#
# ## Actividades:
# 1. Configurar MLflow tracking server
# 2. Registrar experimentos con hiperparámetros
# 3. Guardar métricas y artefactos
# 4. Comparar runs en MLflow UI

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark

# %%
spark = SparkSession.builder \
    .appName("SECOP_MLflow") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %% [markdown]
# ## RETO 1: Configurar MLflow
#
# **Objetivo**: Conectarse al tracking server y crear un experimento.
#
# **Conceptos**:
# - `mlflow.set_tracking_uri()`: URL del servidor MLflow
# - `mlflow.set_experiment()`: Nombre del experimento
# - El tracking server almacena todos los runs, métricas y artefactos
#
# **Instrucciones**:
# 1. Configura la URI del tracking server
# 2. Crea o selecciona un experimento con nombre descriptivo
#
# **Pregunta**: ¿Por qué es importante un tracking server centralizado
# en lugar de guardar métricas en archivos locales?

# %%
# TODO: Configura MLflow
# Pista: El tracking server está en http://mlflow:5000
# mlflow.set_tracking_uri("http://mlflow:5000")

# TODO: Crea un experimento
# experiment_name = "/SECOP_Contratos_Prediccion"
# mlflow.set_experiment(experiment_name)

# print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# print(f"Experimento: {experiment_name}")

# TODO: Responde:
# ¿Qué ventajas tiene un tracking server centralizado?
# Respuesta:

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
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %% [markdown]
# ## RETO 2: Registrar un Experimento Baseline
#
# **Objetivo**: Entrenar un modelo sin regularización y registrarlo en MLflow.
#
# **Estructura de un run MLflow**:
# ```python
# with mlflow.start_run(run_name="nombre"):
#     # 1. Log de parámetros
#     mlflow.log_param("param_name", value)
#
#     # 2. Entrenar modelo
#     model = lr.fit(train)
#
#     # 3. Log de métricas
#     mlflow.log_metric("metric_name", value)
#
#     # 4. Guardar modelo como artefacto
#     mlflow.spark.log_model(model, "model")
# ```
#
# **Instrucciones**:
# 1. Crea un run con nombre descriptivo
# 2. Registra hiperparámetros con `mlflow.log_param()`
# 3. Entrena el modelo y calcula métricas
# 4. Registra métricas con `mlflow.log_metric()`
# 5. Guarda el modelo con `mlflow.spark.log_model()`

# %%
# TODO: Registra tu primer experimento
# with mlflow.start_run(run_name="baseline_no_regularization"):
#     # Hiperparámetros
#     reg_param = 0.0
#     elastic_param = 0.0
#     max_iter = 100
#
#     # TODO: Log de hiperparámetros
#     # mlflow.log_param("regParam", reg_param)
#     # mlflow.log_param("elasticNetParam", elastic_param)
#     # mlflow.log_param("maxIter", max_iter)
#     # mlflow.log_param("model_type", "LinearRegression")
#
#     # TODO: Entrenar modelo
#     # lr = LinearRegression(...)
#     # model = lr.fit(train)
#
#     # TODO: Evaluar
#     # predictions = model.transform(test)
#     # rmse = evaluator.evaluate(predictions)
#
#     # TODO: Log de métricas
#     # mlflow.log_metric("rmse", rmse)
#
#     # TODO: Guardar modelo
#     # mlflow.spark.log_model(model, "model")
#
#     # print(f"RMSE: ${rmse:,.2f}")

# %% [markdown]
# ## RETO 3: Registrar Múltiples Experimentos
#
# **Objetivo**: Entrenar y registrar varios modelos con diferentes regularizaciones
# para comparar en MLflow UI.
#
# **Instrucciones**:
# Crea al menos 3 runs adicionales:
# 1. Ridge (L2): regParam=0.1, elasticNetParam=0.0
# 2. Lasso (L1): regParam=0.1, elasticNetParam=1.0
# 3. ElasticNet: regParam=0.1, elasticNetParam=0.5
#
# **Cada run debe registrar**:
# - Parámetros: regParam, elasticNetParam, maxIter, model_type
# - Métricas: rmse, mae, r2
# - Artefactos: modelo entrenado
#
# **Pregunta**: ¿Por qué registrar múltiples métricas y no solo RMSE?

# %%
# TODO: Implementa los 3 experimentos adicionales
# Pista: Crea una lista de configuraciones y un bucle

# experiments = [
#     {"name": "ridge_l2", "reg": 0.1, "elastic": 0.0, "type": "Ridge"},
#     {"name": "lasso_l1", "reg": 0.1, "elastic": 1.0, "type": "Lasso"},
#     {"name": "elasticnet", "reg": 0.1, "elastic": 0.5, "type": "ElasticNet"},
# ]
#
# for exp in experiments:
#     with mlflow.start_run(run_name=exp["name"]):
#         # TODO: Log parámetros
#         # TODO: Entrenar modelo
#         # TODO: Evaluar con RMSE, MAE y R²
#         # TODO: Log métricas
#         # TODO: Guardar modelo
#         pass

# %% [markdown]
# ## RETO 4: Explorar MLflow UI
#
# **Objetivo**: Usar la interfaz web de MLflow para comparar experimentos.
#
# **Instrucciones**:
# 1. Abre MLflow UI en http://localhost:5000
# 2. Navega al experimento que creaste
# 3. Compara los runs lado a lado
# 4. Ordena por RMSE para encontrar el mejor
# 5. Examina los parámetros y métricas de cada run
#
# **Preguntas**:
# - ¿Qué modelo tiene el menor RMSE?
# - ¿Hay correlación entre regularización y rendimiento?
# - ¿Cómo podrías compartir estos resultados con tu equipo?

# %%
# TODO: Después de explorar la UI, responde:
# Mejor modelo en MLflow UI:
# RMSE del mejor modelo:
# Observaciones:

# %% [markdown]
# ## RETO 5: Agregar Artefactos Personalizados
#
# **Objetivo**: Guardar artefactos adicionales (gráficos, reportes) en un run.
#
# **Instrucciones**:
# 1. Dentro de un run, genera un reporte de métricas en texto
# 2. Guárdalo como artefacto con `mlflow.log_artifact()`
# 3. (Bonus) Genera un gráfico de predicciones vs reales y guárdalo
#
# **Funciones útiles**:
# - `mlflow.log_artifact(local_path)`: Guarda un archivo
# - `mlflow.log_text(text, filename)`: Guarda texto directamente

# %%
# TODO: Crea un run con artefactos personalizados
# with mlflow.start_run(run_name="model_with_artifacts"):
#     # Entrenar modelo (usa los mejores hiperparámetros)
#     # ...
#
#     # TODO: Crear reporte de texto
#     # report = f"""
#     # REPORTE DE MODELO
#     # ==================
#     # RMSE: ${rmse:,.2f}
#     # MAE: ${mae:,.2f}
#     # R²: {r2:.4f}
#     # """
#     # mlflow.log_text(report, "model_report.txt")
#
#     # TODO: (Bonus) Crear y guardar gráfico
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # plt.scatter(labels, predictions)
#     # plt.xlabel("Real")
#     # plt.ylabel("Predicho")
#     # plt.savefig("/tmp/predictions_plot.png")
#     # mlflow.log_artifact("/tmp/predictions_plot.png")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Qué ventajas tiene MLflow sobre guardar métricas en archivos CSV?**
#    Respuesta:
#
# 2. **¿Cómo implementarías MLflow en un proyecto de equipo?**
#    Respuesta:
#
# 3. **¿Qué artefactos adicionales guardarías además del modelo?**
#    Respuesta:
#
# 4. **¿Cómo automatizarías el registro de experimentos?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
print("\n" + "="*60)
print("RESUMEN MLFLOW TRACKING")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Configurado MLflow tracking server")
print("  [ ] Registrado experimento baseline")
print("  [ ] Registrado al menos 3 experimentos adicionales")
print("  [ ] Explorado MLflow UI")
print("  [ ] Comparado métricas entre runs")
print(f"  [ ] Accede a MLflow UI: http://localhost:5000")
print("="*60)

# %%
spark.stop()
