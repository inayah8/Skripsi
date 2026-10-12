import serial
import argparse
import time
import os
import sys
from datetime import datetime

LABELS = {
    "0": "mentah",
    "1": "setengah_matang",
    "2": "matang",
    "3": "terlalu_matang",
    "4": "busuk"
}

def choose_file_interactive():
    print("Masukkan nama file CSV (akan di-append).")
    fname = input("File name (enter untuk 'pisang.csv'): ").strip()
    if fname == "":
        fname = "pisang.csv"
    if os.path.exists(fname):
        print(f"File {fname} exists. Pilih tindakan:")
        print("1) Append (tambah ke file)")
        print("2) Buat file baru (overwrite)")
        choice = input("Pilih 1/2 [1]: ").strip() or "1"
        if choice == "2":
            with open(fname, "w") as f:
                f.write("timestamp_ms,R,G,B,pisang_id,hari_ke,titik,label\n")
            print("File di-reset.\n")
    else:
        with open(fname, "w") as f:
            f.write("timestamp_ms,R,G,B,pisang_id,hari_ke,titik,label\n")
    return fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p")
    parser.add_argument("--baud", "-b", type=int, default=115200)
    parser.add_argument("--file", "-f")
    args = parser.parse_args()

    port = args.port or input("Masukkan serial port (COM3/COM4): ").strip()
    if not port:
        print("Port tidak diberikan.")
        sys.exit(1)

    fname = args.file or choose_file_interactive()

    print(f"Membuka serial {port}, baud {args.baud}")
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except Exception as e:
        print("Gagal buka serial:", e)
        sys.exit(1)

    print("\nSistem siap.")
    print("Tekan tombol ESP32 → 1 kali data terkirim → input info → data disimpan.")
    print("Gunakan Ctrl+C untuk berhenti.\n")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 4:
                print("Data tidak valid:", line)
                continue

            timestamp_ms = parts[0]
            R = parts[1]
            G = parts[2]
            B = parts[3]

            print(f"\nData didapat: R={R} G={G} B={B}")

            pisang_id = input("Masukkan nomor pisang (1–40): ").strip()
            while not pisang_id.isdigit() or not (1 <= int(pisang_id) <= 40):
                pisang_id = input("Nomor tidak valid. Masukkan angka 1–40: ").strip()

            hari_ke = input("Masukkan hari ke- (1–7): ").strip()
            while not hari_ke.isdigit() or not (1 <= int(hari_ke) <= 7):
                hari_ke = input("Hari tidak valid. Masukkan angka 1–7: ").strip()

            print("Pilih titik pengukuran:")
            print("1 = dekat pangkal")
            print("2 = tengah")
            print("3 = ujung")
            titik = input("Titik (1/2/3): ").strip()
            while titik not in ["1", "2", "3"]:
                titik = input("Titik tidak valid. Masukkan 1/2/3: ").strip()

            print("\nMasukkan label pisang:")
            print("0=mentah, 1=setengah matang, 2=matang, 3=terlalu matang, 4=busuk")
            label = input("Label: ").strip()
            while label not in LABELS:
                label = input("Label tidak valid. Masukkan 0/1/2/3/4: ").strip()

            with open(fname, "a") as f:
                f.write(f"{timestamp_ms},{R},{G},{B},{pisang_id},{hari_ke},{titik},{LABELS[label]}\n")

            print(f"✔ Disimpan: Pisang {pisang_id}, Hari {hari_ke}, Titik {titik}, Label {LABELS[label]}\n")

    except KeyboardInterrupt:
        print("\nDihentikan pengguna.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
