# %% [markdown]
# # Notebook 12: Inferencia en Producción
#
# **Sección 16 - MLOps**: Despliegue y predicciones batch
#
# **Objetivo**: Simular un pipeline de producción para generar predicciones
#
# ## Conceptos clave:
# - **Batch Inference**: Predicciones sobre grandes volúmenes de datos
# - **Model Loading**: Cargar modelo desde MLflow Registry
# - **Monitoring**: Verificar calidad de predicciones
# - **Output Formats**: Guardar resultados para consumo (Parquet, CSV)
#
# ## Actividades:
# 1. Cargar modelo desde Model Registry (Production)
# 2. Aplicar transformaciones del pipeline
# 3. Generar predicciones batch sobre datos nuevos
# 4. Monitorear y guardar resultados

# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
import mlflow
import mlflow.spark

# %%
spark = SparkSession.builder \
    .appName("SECOP_Produccion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %% [markdown]
# ## RETO 1: Cargar Modelo en Producción
#
# **Objetivo**: Cargar el modelo registrado en Production desde MLflow.
#
# **Instrucciones**:
# 1. Configura la URI de MLflow
# 2. Define el nombre del modelo y el stage (Production)
# 3. Carga el modelo usando `mlflow.spark.load_model()`
#
# **URI del modelo**: `models:/{nombre_modelo}/{stage}`
#
# **Pregunta**: ¿Por qué cargar desde el Registry en lugar de una ruta de archivo?
# ¿Qué ventajas tiene para un sistema de producción?

# %%
# TODO: Configura MLflow
# mlflow.set_tracking_uri("http://mlflow:5000")

# TODO: Define nombre del modelo y carga desde Registry
# model_name = "secop_prediccion_contratos"
# model_uri = f"models:/{model_name}/Production"

# TODO: Carga el modelo
# print(f"Cargando modelo desde: {model_uri}")
# production_model = mlflow.spark.load_model(model_uri)
# print(f"Modelo cargado: {type(production_model)}")

# TODO: Responde:
# ¿Qué pasaría si no hay modelo en "Production"?
# ¿Cómo manejarías ese error?
# Respuesta:

# %% [markdown]
# ## RETO 2: Cargar y Preparar Datos Nuevos
#
# **Objetivo**: Simular la llegada de datos nuevos para predicción.
#
# **Instrucciones**:
# 1. Carga datos desde el parquet procesado
# 2. Renombra columnas para que coincidan con lo que espera el modelo
# 3. Simula datos "nuevos" eliminando la columna label
#
# **Pregunta**: En un sistema real, ¿de dónde vendrían los datos nuevos?
# - Opciones: Base de datos, API, archivos S3, streaming, etc.

# %%
# TODO: Carga datos para predicción
# df_new = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
# df_new = df_new.withColumnRenamed("features_pca", "features")

# TODO: Simula datos nuevos (sin label)
# df_new_no_label = df_new.drop("valor_del_contrato_num")
# print(f"Contratos para predecir: {df_new_no_label.count():,}")
# print(f"Columnas: {df_new_no_label.columns}")

# %% [markdown]
# ## RETO 3: Generar Predicciones Batch
#
# **Objetivo**: Aplicar el modelo a todos los datos nuevos.
#
# **Instrucciones**:
# 1. Usa `model.transform()` para generar predicciones
# 2. Agrega un timestamp de predicción
# 3. Examina las primeras predicciones
#
# **Pregunta**: ¿Por qué agregar timestamp a las predicciones?
# ¿Qué otros metadatos serían útiles?

# %%
# TODO: Genera predicciones
# predictions_batch = production_model.transform(df_new_no_label)

# TODO: Agrega timestamp
# predictions_batch = predictions_batch.withColumn(
#     "prediction_timestamp", current_timestamp()
# )

# TODO: Muestra las primeras predicciones
# predictions_batch.select("features", "prediction", "prediction_timestamp") \
#     .show(10, truncate=True)

# %% [markdown]
# ## RETO 4: Monitoreo de Predicciones
#
# **Objetivo**: Verificar que las predicciones son razonables.
#
# **Instrucciones**:
# 1. Calcula estadísticas básicas de las predicciones (min, max, avg, std)
# 2. Identifica predicciones negativas (si no tienen sentido en tu contexto)
# 3. Analiza la distribución por rangos
# 4. Define alertas para anomalías
#
# **Pregunta**: ¿Cómo detectarías "data drift" (cambio en la distribución de datos)?
# ¿Qué harías si las predicciones empiezan a ser muy diferentes de lo esperado?

# %%
# TODO: Calcula estadísticas de predicciones
# from pyspark.sql.functions import min as spark_min, max as spark_max, avg, stddev, count
#
# stats = predictions_batch.select(
#     spark_min("prediction").alias("min_pred"),
#     spark_max("prediction").alias("max_pred"),
#     avg("prediction").alias("avg_pred"),
#     stddev("prediction").alias("std_pred"),
#     count("*").alias("total")
# ).collect()[0]
#
# print("=== ESTADÍSTICAS DE PREDICCIONES ===")
# print(f"Total: {stats['total']:,}")
# print(f"Mínimo: ${stats['min_pred']:,.2f}")
# print(f"Máximo: ${stats['max_pred']:,.2f}")
# print(f"Promedio: ${stats['avg_pred']:,.2f}")
# print(f"Std: ${stats['std_pred']:,.2f}")

# %%
# TODO: Analiza la distribución por rangos
# from pyspark.sql.functions import when
#
# prediction_ranges = predictions_batch.select(
#     count(when(col("prediction") < 10000000, True)).alias("< 10M"),
#     count(when((col("prediction") >= 10000000) & (col("prediction") < 100000000), True)).alias("10M-100M"),
#     count(when((col("prediction") >= 100000000) & (col("prediction") < 1000000000), True)).alias("100M-1B"),
#     count(when(col("prediction") >= 1000000000, True)).alias("> 1B")
# )
# prediction_ranges.show()

# TODO: Identifica predicciones anómalas
# ¿Hay predicciones negativas? ¿Valores extremos?
# anomalias = predictions_batch.filter(col("prediction") < 0).count()
# print(f"Predicciones negativas: {anomalias}")

# %% [markdown]
# ## RETO 5: Guardar Resultados
#
# **Objetivo**: Almacenar predicciones en formatos consumibles.
#
# **Instrucciones**:
# 1. Guarda en Parquet (óptimo para analytics y Spark)
# 2. Guarda en CSV (para consumo externo, Excel, etc.)
# 3. Verifica que los archivos se guardaron correctamente
#
# **Pregunta**: ¿Qué formato usarías para cada caso?
# - Dashboard interno: ???
# - Reporte para gerencia: ???
# - Input para otro sistema: ???

# %%
# TODO: Guarda predicciones en Parquet
# predictions_output = "/opt/spark-data/processed/predictions_produccion"
#
# predictions_batch.write.mode("overwrite").parquet(predictions_output + "/parquet")
# print(f"Parquet guardado en: {predictions_output}/parquet")

# TODO: Guarda en CSV
# predictions_batch.select("prediction", "prediction_timestamp") \
#     .write.mode("overwrite") \
#     .option("header", "true") \
#     .csv(predictions_output + "/csv")
# print(f"CSV guardado en: {predictions_output}/csv")

# TODO: Responde:
# ¿Qué formato para cada caso de uso?
# Dashboard:
# Reporte:
# Sistema externo:

# %% [markdown]
# ## RETO 6: Diseñar Pipeline de Producción
#
# **Objetivo**: Pensar en cómo automatizar este proceso.
#
# **Pregunta de diseño**: En un sistema real, este notebook se ejecutaría
# periódicamente. Diseña (en comentarios) cómo lo automatizarías:
#
# 1. **Frecuencia**: ¿Cada hora? ¿Cada día? ¿Bajo demanda?
# 2. **Orquestador**: ¿Airflow? ¿Cron? ¿Spark Streaming?
# 3. **Monitoreo**: ¿Cómo detectas si el modelo se degrada?
# 4. **Reentrenamiento**: ¿Cuándo reentrenar el modelo?
# 5. **Alertas**: ¿Qué condiciones disparan una alerta?

# %%
# TODO: Diseña tu pipeline de producción (en comentarios)
#
# 1. Frecuencia:
#
# 2. Orquestador:
#
# 3. Monitoreo:
#
# 4. Reentrenamiento:
#
# 5. Alertas:

# %% [markdown]
# ## RETO BONUS: Simulación de Scoring Continuo
#
# **Objetivo**: Simular el procesamiento de "lotes" de datos nuevos.
#
# **Instrucciones**:
# 1. Divide los datos en 3 "lotes" simulados
# 2. Para cada lote, genera predicciones y calcula estadísticas
# 3. Compara estadísticas entre lotes
# 4. ¿Las predicciones son consistentes?

# %%
# TODO: Simula scoring por lotes
# batches = df_new_no_label.randomSplit([0.33, 0.33, 0.34], seed=42)
#
# for i, batch in enumerate(batches):
#     preds = production_model.transform(batch)
#     avg_pred = preds.select(avg("prediction")).collect()[0][0]
#     count_pred = preds.count()
#     print(f"Lote {i+1}: {count_pred:,} registros | Predicción promedio: ${avg_pred:,.2f}")

# TODO: ¿Las predicciones son consistentes entre lotes?
# Respuesta:

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Qué pasa si los datos nuevos tienen un esquema diferente al de entrenamiento?**
#    Respuesta:
#
# 2. **¿Cómo implementarías A/B testing en producción?**
#    Respuesta:
#
# 3. **¿Cuándo deberías retirar un modelo de producción?**
#    Respuesta:
#
# 4. **¿Qué métricas de monitoreo son más importantes para tu caso de uso?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
print("\n" + "="*60)
print("RESUMEN INFERENCIA EN PRODUCCIÓN")
print("="*60)
print("Verifica que hayas completado:")
print("  [ ] Modelo cargado desde MLflow Registry")
print("  [ ] Predicciones batch generadas")
print("  [ ] Estadísticas de predicciones calculadas")
print("  [ ] Resultados guardados (Parquet y CSV)")
print("  [ ] Pipeline de producción diseñado")
print("\nPróximos pasos sugeridos:")
print("  1. Configurar monitoreo de data drift")
print("  2. Implementar A/B testing")
print("  3. Automatizar reentrenamiento periódico")
print("  4. Crear alertas de anomalías en predicciones")
print("="*60)

# %%
spark.stop()
