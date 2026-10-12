import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.decomposition import PCA

# ==================== TRAINING ====================

df = pd.read_csv("split_data/train_data.csv")

# ==================== FOLDER OUTPUT ====================

OUTPUT_FOLDER = "Hasil_Train_SVM"

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

# Scaling data untuk SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==================== MODEL SVM ====================

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

svm_model.fit(X_scaled, y_encoded)

# ==================== FEATURE IMPORTANCE ====================

print("\nFEATURE IMPORTANCE (berdasarkan variansi data)")
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.std(X_scaled, axis=0)
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df)
importance_df.to_csv(f"{OUTPUT_FOLDER}/feature_importance.csv", index=False)

# ==================== PREDIKSI DATA TRAINING ====================

y_pred = svm_model.predict(X_scaled)

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
with open(f"{OUTPUT_FOLDER}/ringkasan_statistik.txt", "w", encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("RINGKASAN STATISTIK PER KELAS HASIL PREDIKSI (SVM)\n")
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

with open(f"{OUTPUT_FOLDER}/training_evaluation.txt", "w", encoding='utf-8') as f:
    f.write("HASIL PELATIHAN DAN PREDIKSI (SVM)\n")
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
plt.title("Confusion Matrix (SVM - Hasil Training)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/training_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== DECISION BOUNDARIES (2D PCA) ====================

print("\nMembuat visualisasi decision boundaries...")

# Gunakan PCA untuk reduksi ke 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Buat mesh grid untuk plot
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Prediksi untuk setiap titik di mesh grid
Z = svm_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')

# Plot data points
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=y_encoded, cmap='Set1', 
                     edgecolors='black', linewidth=0.5, alpha=0.8)

# Tambahkan label
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('SVM Decision Boundaries (PCA 2D)')

# Tambahkan legend
legend1 = plt.legend(*scatter.legend_elements(), 
                     title="Kelas", 
                     loc='upper right')
plt.gca().add_artist(legend1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/decision_boundaries_2d.png", dpi=300, bbox_inches='tight')
plt.close()

print("File 'decision_boundaries_2d.png' berhasil disimpan!")

# ==================== SIMPAN MODEL ====================

joblib.dump(svm_model, f"{OUTPUT_FOLDER}/svm_model.pkl")
joblib.dump(label_encoder, f"{OUTPUT_FOLDER}/label_encoder.pkl")
joblib.dump(scaler, f"{OUTPUT_FOLDER}/scaler.pkl")
joblib.dump(pca, f"{OUTPUT_FOLDER}/pca.pkl")

print("\n" + "="*60)
print("TRAINING SVM DAN OUTPUT BERHASIL DIBUAT!")
print("="*60)
print("\nFile yang dihasilkan:")
print("1. statistik_kelas_prediksi.csv     - Statistik rata-rata per kelas")
print("2. detail_prediksi_semua_data.csv   - Detail prediksi tiap sample")
print("3. ringkasan_statistik.txt          - Ringkasan statistik")
print("4. training_evaluation.txt          - Evaluasi model")
print("5. training_confusion_matrix.png    - Confusion matrix")
print("6. decision_boundaries_2d.png       - Decision boundaries 2D (PCA)")
print("7. svm_model.pkl                    - Model SVM")
print("8. label_encoder.pkl                - Label encoder")
print("9. scaler.pkl                       - Standard scaler")
print("10. pca.pkl                         - PCA transformer")
print("\nProses selesai!")