import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Judul aplikasi
st.title("Sistem Informasi Clustering dan Prediksi Kasus COVID-19 di Indonesia")

# =========================
# DATA DUMMY
# =========================
data = {
    'Provinsi': ['Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Bali'],
    'Total_Kasus': [150000, 120000, 110000, 100000, 50000],
    'Sembuh': [140000, 115000, 105000, 95000, 48000],
    'Meninggal': [5000, 4000, 4500, 4200, 1000]
}

df = pd.DataFrame(data)

st.subheader("Data Kasus COVID-19 (Contoh)")
st.write(df)

# =========================
# CLUSTERING
# =========================
st.subheader("Clustering Provinsi")
fitur = df[['Total_Kasus', 'Sembuh', 'Meninggal']]

k = st.slider("Jumlah Klaster", 2, 5, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(fitur)

# Tambahkan label zona berdasarkan klaster
zona_mapping = {
    0: "Zona Merah",
    1: "Zona Kuning",
    2: "Zona Hijau",
    3: "Zona Oranye",
    4: "Zona Biru"
}
df['Zona'] = df['Cluster'].map(zona_mapping)

st.write("Hasil Clustering dan Keterangan Zona:")
st.write(df[['Provinsi', 'Total_Kasus', 'Sembuh', 'Meninggal', 'Cluster', 'Zona']])

# =========================
# REGRESI LINIER
# =========================
st.subheader("Prediksi Kasus dengan Regresi Linier")
selected_prov = st.selectbox("Pilih Provinsi untuk Prediksi", df['Provinsi'])

# Simulasi data tren harian (misalnya 10 hari terakhir)
kasus_awal = st.slider("Jumlah Kasus Hari ke-0", 1000, 20000, 10000)
trend = np.array([kasus_awal + i*100 for i in range(10)])
hari = np.arange(10).reshape(-1, 1)

model = LinearRegression()
model.fit(hari, trend)

hari_pred = np.arange(10, 17).reshape(-1, 1)
prediksi = model.predict(hari_pred)

st.write(f"Prediksi Kasus 7 Hari ke Depan untuk {selected_prov}:")
pred_df = pd.DataFrame({"Hari ke-": np.arange(10, 17), "Prediksi Kasus": prediksi.astype(int)})
st.write(pred_df)