import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array
from sklearn.ensemble import RandomForestClassifier

# ================================
# ✅ START SPARK
# ================================
spark = SparkSession.builder \
    .appName("ExportModel") \
    .getOrCreate()

print("✅ Spark started")

# ================================
# ✅ LOAD FEATURE ENGINEERED DATA
# ================================
df = spark.read.parquet("/output/ieee_fraud_features")
print("✅ Features loaded")

# ================================
# ✅ VECTOR → ARRAY (PROPER WAY)
# ================================
df = df.select(
    col("label"),
    vector_to_array(col("features")).alias("features_arr")
)

# ================================
# ✅ CONVERT TO PANDAS SAFELY
# ================================
pdf = df.toPandas()

X = np.array(pdf["features_arr"].to_list(), dtype=np.float32)
y = pdf["label"].astype(int).values

print("✅ Data converted to NumPy")

# ================================
# ✅ TRAIN SCIKIT-LEARN MODEL
# ================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

print("✅ Model trained")

# ================================
# ✅ SAVE MODEL FOR STREAMLIT
# ================================
joblib.dump(model, "/output/ieee_fraud_model/model.pkl")

print("✅ sklearn model.pkl exported successfully!")

spark.stop()
