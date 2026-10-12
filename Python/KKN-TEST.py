import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

# ==================== LOAD MODEL ====================
print("Memuat model KNN yang sudah dilatih...")
try:
    # Mengambil model dari folder Hasil_Train_KNN
    knn_model = joblib.load("Hasil_Train_KNN/knn_model.pkl")
    label_encoder = joblib.load("Hasil_Train_KNN/label_encoder.pkl")
    scaler = joblib.load("Hasil_Train_KNN/scaler.pkl")
    pca = joblib.load("Hasil_Train_KNN/pca.pkl")
    print("Model berhasil dimuat!")
    print(f"Kelas yang tersedia: {list(label_encoder.classes_)}")
except FileNotFoundError:
    print("ERROR: File model tidak ditemukan!")
    print("Pastikan Anda sudah menjalankan training KNN terlebih dahulu.")
    exit()

# ==================== LOAD DATA UJI ====================
print("\nMemuat data uji...")
try:
    # Menggunakan path yang sama dengan RF
    df_test = pd.read_csv("split_data/test_data.csv")
    print(f"Data uji berhasil dimuat! Jumlah data: {len(df_test)}")
except FileNotFoundError:
    print("ERROR: File test_data.csv tidak ditemukan!")
    print("Pastikan Anda sudah menjalankan split data terlebih dahulu.")
    exit()

# ==================== FOLDER OUTPUT ====================

OUTPUT_FOLDER = "Hasil_Test_KNN"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Folder output: {OUTPUT_FOLDER}")

# ==================== PREPROCESSING DATA UJI ====================
print("\nMelakukan preprocessing data uji...")

# Konversi kolom RGB ke numeric
for col in ["fR", "fG", "fB"]:
    df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

df_test_clean = df_test.dropna(subset=["fR", "fG", "fB"]).copy()

if len(df_test_clean) == 0:
    print("ERROR: Tidak ada data valid setelah preprocessing!")
    exit()

print(f"Data valid setelah cleaning: {len(df_test_clean)} baris")
print(f"Data dibuang: {len(df_test) - len(df_test_clean)} baris")

X_test = df_test_clean[
    [
        "fR",
        "fG",
        "fB"
    ]
]

# Scaling data uji menggunakan scaler yang sudah dilatih
X_test_scaled = scaler.transform(X_test)

# Cek apakah label tersedia
if "label" in df_test_clean.columns:
    y_test_available = True
    y_test = df_test_clean["label"]
    print("Label testing tersedia, akan dilakukan evaluasi.")
else:
    y_test_available = False
    print("WARNING: Kolom 'label' tidak ditemukan. Akan menggunakan prediksi saja.")

# ==================== PREDIKSI ====================
print("\nMelakukan prediksi...")

# Prediksi semua data
y_pred_encoded = knn_model.predict(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Prediksi probabilitas (untuk KNN, probabilitas berdasarkan jarak)
y_pred_proba = knn_model.predict_proba(X_test_scaled)

# Tambahkan hasil prediksi ke dataframe
df_test_clean["predicted_label"] = y_pred_labels
df_test_clean["prediction_confidence"] = [max(proba) for proba in y_pred_proba]

# Tambahkan semua kelas dengan probabilitasnya
for i, class_name in enumerate(label_encoder.classes_):
    df_test_clean[f"prob_{class_name}"] = y_pred_proba[:, i]

# Evaluasi jika label aktual tersedia
if y_test_available:
    # Encode label aktual
    try:
        y_test_encoded = label_encoder.transform(y_test)
        df_test_clean["actual_label"] = y_test
        df_test_clean["prediction_correct"] = df_test_clean["actual_label"] == df_test_clean["predicted_label"]
        
        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(y_test_encoded, y_pred_encoded, average="weighted", zero_division=0)
        recall = recall_score(y_test_encoded, y_pred_encoded, average="weighted", zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred_encoded, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        print("Evaluasi berhasil dilakukan!")
    except ValueError as e:
        print(f"WARNING: Terdapat label baru di data uji: {e}")
        print("Akan melanjutkan dengan prediksi saja.")
        y_test_available = False

# ==================== OUTPUT STATISTIK PER KELAS ====================
print("\nMembuat statistik per kelas prediksi...")

# Buat dataframe untuk statistik per kelas yang diprediksi
class_stats_test = []

for kelas in label_encoder.classes_:
    # Filter data yang diprediksi sebagai kelas ini
    mask = df_test_clean["predicted_label"] == kelas
    kelas_data = df_test_clean[mask]
    
    if len(kelas_data) > 0:
        # Hitung statistik
        total_samples = len(kelas_data)
        
        # Akurasi per kelas (jika label tersedia)
        if y_test_available:
            accuracy_class = len(kelas_data[kelas_data["prediction_correct"] == True]) / total_samples * 100
        else:
            accuracy_class = None
        
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
        
        # Rata-rata confidence
        avg_confidence = kelas_data["prediction_confidence"].mean() * 100
        
        class_stats_test.append({
            "Kelas": kelas,
            "Jumlah_Sample": total_samples,
            "Persentase_Data": total_samples / len(df_test_clean) * 100,
            "Akurasi_Kelas(%)": accuracy_class if y_test_available else None,
            "Confidence_Rata2(%)": avg_confidence,
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
stats_test_df = pd.DataFrame(class_stats_test)

# ==================== OUTPUT DETAIL SETIAP SAMPLE ====================
print("\nMenyiapkan output detail...")

# Menangani kasus ketika actual_label tidak tersedia
if y_test_available:
    output_detail = df_test_clean[
        ["fR", "fG", "fB",
         "actual_label",
         "predicted_label",
         "prediction_correct",
         "prediction_confidence"]
    ]
else:
    output_detail = df_test_clean[
        ["fR", "fG", "fB",
         "predicted_label",
         "prediction_confidence"]
    ]

# ==================== SIMPAN FILE OUTPUT ====================
print("\nMenyimpan file output...")

# 1. File statistik per kelas
stats_test_df.to_csv(f"{OUTPUT_FOLDER}/statistik_prediksi_data_uji.csv", index=False)
print("✓ File 'statistik_prediksi_data_uji.csv' berhasil disimpan!")

# 2. File detail semua data dengan prediksi
output_detail.to_csv(f"{OUTPUT_FOLDER}/detail_prediksi_data_uji.csv", index=False)
print("✓ File 'detail_prediksi_data_uji.csv' berhasil disimpan!")

# 3. File ringkasan statistik
with open(f"{OUTPUT_FOLDER}/ringkasan_prediksi_data_uji.txt", "w", encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("RINGKASAN STATISTIK PER KELAS HASIL PREDIKSI (KNN - DATA UJI)\n")
    f.write("="*60 + "\n\n")
    
    for idx, row in stats_test_df.iterrows():
        f.write(f"KELAS: {row['Kelas']}\n")
        f.write(f"  Jumlah sample: {row['Jumlah_Sample']}\n")
        f.write(f"  Persentase data: {row['Persentase_Data']:.1f}%\n")
        f.write(f"  Confidence rata-rata: {row['Confidence_Rata2(%)']:.1f}%\n")
        if y_test_available and row['Akurasi_Kelas(%)'] is not None:
            f.write(f"  Akurasi prediksi: {row['Akurasi_Kelas(%)']:.2f}%\n")
        f.write(f"  Rata-rata RGB: ({row['R_Rata2']:.1f}, {row['G_Rata2']:.1f}, {row['B_Rata2']:.1f})\n")
        f.write(f"  Standar deviasi RGB: ({row['R_Std']:.1f}, {row['G_Std']:.1f}, {row['B_Std']:.1f})\n")
        f.write(f"  Rentang R: {row['R_Min']:.0f}-{row['R_Max']:.0f}\n")
        f.write(f"  Rentang G: {row['G_Min']:.0f}-{row['G_Max']:.0f}\n")
        f.write(f"  Rentang B: {row['B_Min']:.0f}-{row['B_Max']:.0f}\n")
        f.write("-"*40 + "\n")
    
    # Total statistik
    f.write("\n" + "="*60 + "\n")
    f.write("TOTAL STATISTIK\n")
    f.write("="*60 + "\n")
    f.write(f"Total data uji (raw): {len(df_test)}\n")
    f.write(f"Total data valid: {len(df_test_clean)}\n")
    f.write(f"Data dibuang: {len(df_test) - len(df_test_clean)}\n")
    
    if y_test_available:
        f.write(f"Akurasi overall: {accuracy*100:.2f}%\n")
    
    # Distribusi kelas
    f.write("\nDistribusi Kelas Prediksi:\n")
    distribusi = df_test_clean['predicted_label'].value_counts()
    for label, count in distribusi.items():
        persentase = count / len(df_test_clean) * 100
        f.write(f"  {label}: {count} sample ({persentase:.1f}%)\n")

print("✓ File 'ringkasan_prediksi_data_uji.txt' berhasil disimpan!")

# ==================== EVALUASI MODEL ====================
if y_test_available:
    print("\nMenyimpan evaluasi model...")
    
    with open(f"{OUTPUT_FOLDER}/testing_evaluation.txt", "w", encoding='utf-8') as f:
        f.write("HASIL EVALUASI MODEL KNN PADA DATA UJI\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n\n")
        f.write(classification_report(
            y_test_encoded, y_pred_encoded,
            target_names=label_encoder.classes_,
            zero_division=0
        ))
    
    print("✓ File 'testing_evaluation.txt' berhasil disimpan!")
    
    # Tampilkan di terminal
    print("\n" + "="*60)
    with open(f"{OUTPUT_FOLDER}/testing_evaluation.txt", "r") as f:
        print(f.read())

# ==================== CONFUSION MATRIX ====================
if y_test_available:
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
    plt.title("Confusion Matrix (KNN - Data Uji)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/testing_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ File 'testing_confusion_matrix.png' berhasil disimpan!")

# ==================== DECISION BOUNDARIES DATA UJI (2D PCA) ====================
print("\nMembuat decision boundaries untuk data uji...")

# Transform data uji dengan PCA
X_test_pca = pca.transform(X_test_scaled)

# Buat mesh grid untuk plot
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Prediksi untuk setiap titik di mesh grid
Z = knn_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')

# Plot data points
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                     c=y_pred_encoded, cmap='Set1', 
                     edgecolors='black', linewidth=0.5, alpha=0.8)

# Tambahkan label
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('KNN Decision Boundaries - Data Uji (PCA 2D)')

# Tambahkan legend
legend1 = plt.legend(*scatter.legend_elements(), 
                     title="Kelas Prediksi", 
                     loc='upper right')
plt.gca().add_artist(legend1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/decision_boundaries_data_uji.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ File 'decision_boundaries_data_uji.png' berhasil disimpan!")

# ==================== OUTPUT TERMINAL ====================
print("\n" + "="*60)
print("TESTING DATA KNN DAN OUTPUT BERHASIL DIBUAT!")
print("="*60)

if y_test_available:
    print(f"\nMETRIK EVALUASI:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")

print(f"\nSTATISTIK DATA:")
print(f"  Total data uji (raw): {len(df_test)}")
print(f"  Data valid          : {len(df_test_clean)}")
print(f"  Data dibuang        : {len(df_test) - len(df_test_clean)}")
print(f"  Jumlah kelas        : {len(label_encoder.classes_)}")

print("\nDISTRIBUSI PREDIKSI:")
for _, row in stats_test_df.iterrows():
    print(f"  {row['Kelas']}: {row['Jumlah_Sample']} sample ({row['Persentase_Data']:.1f}%)")

print("\nFILE OUTPUT YANG DIHASILKAN:")
print("1. statistik_prediksi_data_uji.csv     - Statistik rata-rata per kelas")
print("2. detail_prediksi_data_uji.csv        - Detail prediksi tiap sample")
print("3. ringkasan_prediksi_data_uji.txt     - Ringkasan statistik")
if y_test_available:
    print("4. testing_evaluation.txt             - Evaluasi model pada data uji")
    print("5. testing_confusion_matrix.png       - Confusion matrix data uji")
print("6. decision_boundaries_data_uji.png    - Decision boundaries data uji (PCA 2D)")

print("\nProses testing KNN selesai!")