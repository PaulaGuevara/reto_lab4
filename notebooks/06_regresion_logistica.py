# %% [markdown]
# # Notebook 06: Regresión Logística para Clasificación
#
# **Sección 14**: Clasificación Binaria
#
# **Objetivo**: Clasificar contratos según riesgo de incumplimiento
#
# ## RETO PRINCIPAL: Crear tu propia variable objetivo
#
# **Problema**: El dataset no tiene una columna de "riesgo de incumplimiento".
# ¡TENDRÁS QUE CREARLA!
#
# **Instrucciones**:
# Define un criterio para clasificar contratos como "alto riesgo" (1) o "bajo riesgo" (0)
#
# **Posibles criterios**:
# - Contratos con valor > percentil 90
# - Contratos con duración > 365 días
# - Contratos de ciertos departamentos
# - Combinación de múltiples factores
#
# **TU DECISIÓN**: ¿Qué define un contrato de alto riesgo?

# %%
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when

# %%
spark = SparkSession.builder \
    .appName("SECOP_RegresionLogistica") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")
print(f"Registros: {df.count():,}")

# %% [markdown]
# ## RETO 1: Crear Variable Objetivo Binaria
#
# **Objetivo**: Crear columna "riesgo" con valores 0 (bajo) o 1 (alto)
#
# **Ejemplos de criterios**:
# - Contratos grandes: valor > percentil 90
# - Contratos urgentes: plazo < 30 días
# - Departamentos históricos de incumplimiento
# - Combinación de factores

# %%
# TODO: Calcula el percentil 90 del valor de contratos
# Pista: df.approxQuantile("valor_del_contrato_num", [0.9], 0.01)

# TODO: Define tu criterio de riesgo
# Ejemplo básico (PERSONALIZA ESTO):
df = df.withColumn(
    "riesgo",
    when(
        # TODO: Completa tu condición aquí
        # Ejemplo: (col("valor_del_contrato_num") > threshold) | (col("plazo") < 30)
        col("valor_del_contrato_num") > 1000000000,  # Placeholder
        1
    ).otherwise(0)
)

# Justifica tu decisión:
# Criterio elegido:
# Razón:

# %% [markdown]
# ## RETO 2: Balance de Clases
#
# **Problema crítico**: Si tienes 95% clase 0 y 5% clase 1, tu modelo
# puede lograr 95% accuracy simplemente prediciendo SIEMPRE clase 0.
#
# **Pregunta**: ¿Tu dataset está balanceado?

# %%
# TODO: Analiza la distribución de clases
print("\n=== DISTRIBUCIÓN DE CLASES ===")
class_distribution = df.groupBy("riesgo").count()
class_distribution.show()

# TODO: Calcula porcentajes
total = df.count()
clase_0 = df.filter(col("riesgo") == 0).count()
clase_1 = df.filter(col("riesgo") == 1).count()

print(f"Clase 0 (Bajo riesgo): {clase_0:,} ({clase_0/total*100:.1f}%)")
print(f"Clase 1 (Alto riesgo): {clase_1:,} ({clase_1/total*100:.1f}%)")

# TODO: Responde:
# ¿Está balanceado? (Sí/No):
# Si NO, ¿qué harías?
# Opciones:
# A) Undersample clase mayoritaria
# B) Oversample clase minoritaria (duplicar registros)
# C) Usar class_weight en el modelo
# D) Cambiar el threshold de clasificación

# Tu decisión:

# %% [markdown]
# ## PASO 1: Preparar Datos

# %%
# Renombrar columnas para el modelo
df_binary = df.withColumnRenamed("riesgo", "label") \
               .withColumnRenamed("features_raw", "features")

# Filtrar nulos
df_binary = df_binary.filter(col("label").isNotNull() & col("features").isNotNull())

# Split train/test
train, test = df_binary.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %% [markdown]
# ## RETO 3: Entender la Regresión Logística
#
# **Pregunta conceptual**: ¿En qué se diferencia de regresión lineal?
#
# **Opciones**:
# - A) Predice probabilidades entre 0 y 1
# - B) Usa función sigmoid
# - C) Es para clasificación, no para valores continuos
# - D) Todas las anteriores
#
# **Responde y explica**

# %%
# TODO: Escribe tu respuesta
# La regresión logística:

# %% [markdown]
# ## RETO 4: Configurar el Modelo
#
# **Parámetros clave**:
# - maxIter: Iteraciones de optimización
# - regParam: Regularización (prevenir overfitting)
# - threshold: ¿A partir de qué probabilidad clasificar como 1?
# - family: "binomial" (predeterminado para clasificación binaria)

# %%
# TODO: Configura el modelo
lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.0,  # TODO: ¿Deberías usar regularización?
    threshold=0.5  # TODO: ¿Es apropiado si las clases están desbalanceadas?
)

print("✓ Clasificador configurado")

# TODO: Responde:
# Si tienes 90% clase 0 y 10% clase 1, ¿qué threshold usarías?
# Respuesta:

# %%
# Entrenar modelo
print("\nEntrenando clasificador...")
lr_model = lr_classifier.fit(train)
print("✓ Modelo entrenado")

# %% [markdown]
# ## PASO 2: Predicciones

# %%
predictions = lr_model.transform(test)

print("\n=== PRIMERAS PREDICCIONES ===")
predictions.select("label", "prediction", "probability").show(10, truncate=False)

# %% [markdown]
# ## RETO 5: Interpretar Probabilidades
#
# **Pregunta**: Si ves probability=[0.8, 0.2], ¿qué significa?
#
# **Opciones**:
# - A) 80% chance de clase 0, 20% de clase 1
# - B) 20% chance de clase 0, 80% de clase 1
# - C) El valor está entre 0.8 y 0.2
#
# **Verifica en las predicciones de arriba**

# %%
# TODO: Analiza casos donde el modelo está "inseguro"
# Filtra predicciones donde la probabilidad esté entre 0.4 y 0.6

# predictions.filter(
#     (col("probability")[1] > 0.4) & (col("probability")[1] < 0.6)
# ).select("label", "prediction", "probability").show(10, truncate=False)

# TODO: ¿Cuántos casos "dudosos" hay?

# %% [markdown]
# ## RETO 6: Evaluación con Múltiples Métricas
#
# **Concepto**: Para clasificación, accuracy NO es suficiente
#
# **Métricas importantes**:
# - **AUC-ROC**: Área bajo la curva ROC (0.5 = random, 1.0 = perfecto)
# - **Precision**: De los predichos como positivos, ¿cuántos lo son?
# - **Recall**: De los positivos reales, ¿cuántos detectamos?
# - **F1-Score**: Balance entre precision y recall

# %%
# TODO: Calcula AUC-ROC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator_auc.evaluate(predictions)

# TODO: Calcula métricas multiclase
evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

print("\n" + "="*60)
print("MÉTRICAS DE CLASIFICACIÓN")
print("="*60)
print(f"AUC-ROC:   {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

# TODO: Interpreta:
# ¿Es bueno un AUC de 0.75?
# Respuesta:

# %% [markdown]
# ## RETO 7: Matriz de Confusión
#
# **Objetivo**: Entender DÓNDE falla el modelo
#
# **Conceptos**:
# - **TP (True Positive)**: Predijo 1, era 1 ✓
# - **TN (True Negative)**: Predijo 0, era 0 ✓
# - **FP (False Positive)**: Predijo 1, era 0 ✗ (Falsa alarma)
# - **FN (False Negative)**: Predijo 0, era 1 ✗ (Miss crítico)

# %%
# TODO: Construye la matriz de confusión
print("\n=== MATRIZ DE CONFUSIÓN ===")
confusion_matrix = predictions.groupBy("label", "prediction").count()
confusion_matrix.orderBy("label", "prediction").show()

# TODO: Calcula manualmente cada valor
# TP = df.filter((col("label") == 1) & (col("prediction") == 1)).count()
# TN = ...
# FP = ...
# FN = ...

# TODO: Responde para ESTE problema específico:
# ¿Qué es peor?
# - Falso Positivo (predecir alto riesgo cuando es bajo)
#   Consecuencia:
# - Falso Negativo (predecir bajo riesgo cuando es alto)
#   Consecuencia:

# Tu respuesta:

# %% [markdown]
# ## RETO BONUS 1: Ajustar Threshold
#
# **Objetivo**: Optimizar el threshold según tu prioridad
#
# **Escenario**: Si quieres detectar TODOS los casos de alto riesgo,
# incluso aumentando falsos positivos, debes BAJAR el threshold

# %%
# TODO: Experimenta con diferentes thresholds
thresholds = [0.3, 0.5, 0.7]

print("\n=== COMPARACIÓN DE THRESHOLDS ===")
for t in thresholds:
    lr_temp = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        threshold=t
    )
    model_temp = lr_temp.fit(train)
    preds_temp = model_temp.transform(test)

    # Evaluar
    acc_temp = evaluator_multi.evaluate(preds_temp, {evaluator_multi.metricName: "accuracy"})
    rec_temp = evaluator_multi.evaluate(preds_temp, {evaluator_multi.metricName: "weightedRecall"})

    print(f"Threshold={t}: Accuracy={acc_temp:.3f}, Recall={rec_temp:.3f}")

# TODO: ¿Qué threshold elegirías? ¿Por qué?
# Respuesta:

# %% [markdown]
# ## RETO BONUS 2: Curva ROC
#
# **Objetivo**: Visualizar el trade-off entre TPR y FPR
#
# **Conceptos**:
# - TPR (True Positive Rate) = Recall = TP / (TP + FN)
# - FPR (False Positive Rate) = FP / (FP + TN)

# %%
# TODO: (Avanzado) Implementa la curva ROC
# Necesitarás:
# 1. Extraer probabilidades del modelo
# 2. Para cada threshold posible, calcular TPR y FPR
# 3. Graficar TPR vs FPR

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Extraer probabilidades
# prob_df = predictions.select("label", "probability").toPandas()
# probs = np.array([p[1] for p in prob_df['probability']])
# labels = prob_df['label'].values
#
# # Calcular TPR y FPR para diferentes thresholds
# thresholds_roc = np.linspace(0, 1, 100)
# tpr_list = []
# fpr_list = []
#
# for t in thresholds_roc:
#     y_pred = (probs >= t).astype(int)
#     tp = np.sum((y_pred == 1) & (labels == 1))
#     fp = np.sum((y_pred == 1) & (labels == 0))
#     tn = np.sum((y_pred == 0) & (labels == 0))
#     fn = np.sum((y_pred == 0) & (labels == 1))
#
#     tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#
#     tpr_list.append(tpr)
#     fpr_list.append(fpr)
#
# # Graficar
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_list, tpr_list, label=f'AUC = {auc:.3f}')
# plt.plot([0, 1], [0, 1], 'r--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Curva ROC')
# plt.legend()
# plt.grid(True)
# plt.savefig('/opt/spark-data/processed/roc_curve.png')
# print("Curva ROC guardada")

# %% [markdown]
# ## Preguntas de Reflexión
#
# 1. **¿Cuándo usarías regresión logística vs árboles de decisión?**
#    Respuesta:
#
# 2. **¿Qué significa un AUC de 0.5?**
#    Respuesta:
#
# 3. **¿Cómo manejarías un dataset con 99% clase 0 y 1% clase 1?**
#    Respuesta:
#
# 4. **¿Por qué accuracy puede ser engañoso en clasificación desbalanceada?**
#    Respuesta:

# %%
# TODO: Escribe tus respuestas arriba

# %%
# Guardar modelo
model_path = "/opt/spark-data/processed/logistic_regression_model"
lr_model.write().overwrite().save(model_path)
print(f"\n✓ Modelo guardado en: {model_path}")

# %%
print("\n" + "="*60)
print("RESUMEN CLASIFICACIÓN")
print("="*60)
print(f"✓ Criterio de riesgo definido")
print(f"✓ Modelo entrenado")
print(f"✓ AUC-ROC: {auc:.4f}")
print(f"✓ F1-Score: {f1:.4f}")
print(f"✓ Próximo paso: Regularización (notebook 07)")
print("="*60)

# %%
spark.stop()
