import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_predict

# ==================== TRAINING ====================

# PERBAIKAN: Menggunakan path yang benar dari folder split_data
df = pd.read_csv("split_data/train_data.csv")

# ==================== FOLDER OUTPUT ====================

OUTPUT_FOLDER = "Hasil_Train"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Folder output: {OUTPUT_FOLDER}")

for col in ["fR", "fG", "fB"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Buang baris non-RGB (log ESP32)
df = df.dropna(subset=["fR","fG","fB"])

# ==================== 

X = df[
    [
        "fR",
        "fG",
        "fB"
    ]
]

y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y_encoded)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df)

importance_df.to_csv(
    f"{OUTPUT_FOLDER}/feature_importance.csv",
    index=False
)

# Prediksi semua data training
y_pred = rf.predict(X)

# Decode prediksi ke label asli
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Tambahkan prediksi ke dataframe
df["predicted_label"] = y_pred_labels
df["prediction_correct"] = df["label"] == df["predicted_label"]

# ==================== OUTPUT STATISTIK PER KELAS ====================

# Buat dataframe untuk statistik per kelas yang diprediksi
class_stats = []

for kelas in label_encoder.classes_:
    # Filter data yang diprediksi sebagai kelas ini
    mask = df["predicted_label"] == kelas
    kelas_data = df[mask]
    
    if len(kelas_data) > 0:
        # Hitung statistik
        total_samples = len(kelas_data)
        accuracy_class = len(kelas_data[kelas_data["prediction_correct"] == True]) / total_samples * 100
        
        # Rata-rata RGB untuk kelas ini
        avg_r = kelas_data["fR"].mean()
        avg_g = kelas_data["fG"].mean()
        avg_b = kelas_data["fB"].mean()
        
        # Standar deviasi
        std_r = kelas_data["fR"].std()
        std_g = kelas_data["fG"].std()
        std_b = kelas_data["fB"].std()
        
        # Rentang nilai
        min_r = kelas_data["fR"].min()
        max_r = kelas_data["fR"].max()
        min_g = kelas_data["fG"].min()
        max_g = kelas_data["fG"].max()
        min_b = kelas_data["fB"].min()
        max_b = kelas_data["fB"].max()
        
        class_stats.append({
            "Kelas": kelas,
            "Jumlah_Sample": total_samples,
            "Akurasi_Kelas(%)": accuracy_class,
            "R_Rata2": avg_r,
            "G_Rata2": avg_g,
            "B_Rata2": avg_b,
            "R_Std": std_r,
            "G_Std": std_g,
            "B_Std": std_b,
            "R_Min": min_r,
            "R_Max": max_r,
            "G_Min": min_g,
            "G_Max": max_g,
            "B_Min": min_b,
            "B_Max": max_b
        })

# Konversi ke DataFrame
stats_df = pd.DataFrame(class_stats)

# ==================== OUTPUT DETAIL SETIAP SAMPLE ====================

# Buat file CSV dengan semua data + prediksi
output_detail = df[
    ["fR", "fG", "fB",
     "label",
     "predicted_label",
     "prediction_correct"]
]

# ==================== SIMPAN FILE OUTPUT ====================

# 1. File statistik per kelas
stats_df.to_csv(f"{OUTPUT_FOLDER}/statistik_kelas_prediksi.csv", index=False)
print("File 'statistik_kelas_prediksi.csv' berhasil disimpan!")

# 2. File detail semua data dengan prediksi
output_detail.to_csv(f"{OUTPUT_FOLDER}/detail_prediksi_semua_data.csv", index=False)
print("File 'detail_prediksi_semua_data.csv' berhasil disimpan!")

# 3. File ringkasan statistik
with open(f"{OUTPUT_FOLDER}/ringkasan_statistik.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("RINGKASAN STATISTIK PER KELAS HASIL PREDIKSI\n")
    f.write("="*60 + "\n\n")
    
    for idx, row in stats_df.iterrows():
        f.write(f"KELAS: {row['Kelas']}\n")
        f.write(f"  Jumlah sample: {row['Jumlah_Sample']}\n")
        f.write(f"  Akurasi prediksi: {row['Akurasi_Kelas(%)']:.2f}%\n")
        f.write(f"  Rata-rata RGB: ({row['R_Rata2']:.1f}, {row['G_Rata2']:.1f}, {row['B_Rata2']:.1f})\n")
        f.write(f"  Rentang R: {row['R_Min']:.0f}-{row['R_Max']:.0f}\n")
        f.write(f"  Rentang G: {row['G_Min']:.0f}-{row['G_Max']:.0f}\n")
        f.write(f"  Rentang B: {row['B_Min']:.0f}-{row['B_Max']:.0f}\n")
        f.write("-"*40 + "\n")
    
    # Total statistik
    f.write("\n" + "="*60 + "\n")
    f.write("TOTAL STATISTIK\n")
    f.write("="*60 + "\n")
    f.write(f"Total sample: {len(df)}\n")
    overall_accuracy = len(df[df['prediction_correct'] == True]) / len(df) * 100
    f.write(f"Akurasi overall: {overall_accuracy:.2f}%\n")
    
    # Distribusi kelas
    f.write("\nDistribusi Kelas Prediksi:\n")
    distribusi = df['predicted_label'].value_counts()
    for label, count in distribusi.items():
        persentase = count / len(df) * 100
        f.write(f"  {label}: {count} sample ({persentase:.1f}%)\n")

print("File 'ringkasan_statistik.txt' berhasil disimpan!")

# ==================== EVALUASI MODEL ====================

accuracy = accuracy_score(y_encoded, y_pred)
precision = precision_score(y_encoded, y_pred, average="weighted")
recall = recall_score(y_encoded, y_pred, average="weighted")
f1 = f1_score(y_encoded, y_pred, average="weighted")
cm = confusion_matrix(y_encoded, y_pred)

with open(f"{OUTPUT_FOLDER}/training_evaluation.txt", "w") as f:
    f.write("HASIL PELATIHAN DAN PREDIKSI\n")
    f.write("="*60 + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write(classification_report(
        y_encoded, y_pred,
        target_names=label_encoder.classes_
    ))
    
with open(f"{OUTPUT_FOLDER}/training_evaluation.txt", "r") as f:
    print(f.read())

# Confusion Matrix
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Hasil Training)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/training_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ==================== SIMPAN MODEL ====================

joblib.dump(rf, f"{OUTPUT_FOLDER}/rf_model.pkl")
joblib.dump(label_encoder, f"{OUTPUT_FOLDER}/label_encoder.pkl")

print("\n" + "="*60)
print("TRAINING DAN OUTPUT BERHASIL DIBUAT!")
print("="*60)
print("\nFile yang dihasilkan:")
print("1. statistik_kelas_prediksi.csv     - Statistik rata-rata per kelas")
print("2. detail_prediksi_semua_data.csv   - Detail prediksi tiap sample")
print("3. ringkasan_statistik.txt          - Ringkasan statistik")
print("4. training_evaluation.txt          - Evaluasi model")
print("5. training_confusion_matrix.png    - Confusion matrix")
print("6. rf_model.pkl                     - Model Random Forest")
print("7. label_encoder.pkl                - Label encoder")
print("\nProses selesai!")