import streamlit as st
import joblib

# Load model dan vectorizer yang sudah dilatih (pastikan file .pkl satu folder)
kmeans_model = joblib.load('kmeans_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Mapping label cluster sesuai hasil analisis riset lu sebelumnya
cluster_names = {
    0: "Medis & Keperawatan Klinis",
    1: "Manajemen & Administrasi Kesehatan",
    2: "Hospitality & Food and Beverage",
    3: "Teknologi, Data & Rekayasa",
    4: "Ritel, Penjualan & Layanan Pelanggan"
}

st.title("TalentSync AI 🚀")
st.write("Sistem Cerdas Pemetaan Keahlian Kerja & Sektor Industri")

# Kotak input untuk user
user_skills = st.text_input("Masukkan daftar skill (pisahkan dengan koma):", "python, sql, project management")

# Tombol eksekusi
if st.button("Analisis Sektor"):
    if user_skills:
        # Ubah teks input menjadi vektor matematika
        skills_vector = vectorizer.transform([user_skills])
        
        # Minta AI memprediksi cluster
        predicted_cluster = kmeans_model.predict(skills_vector)[0]
        industry_name = cluster_names.get(predicted_cluster, "Sektor Tidak Terdefinisi")
        
        st.success(f"Berdasarkan irisan kompetensi, profil keahlian ini mendominasi sektor: **{industry_name}**")
    else:
        st.warning("Harap masukkan daftar skill terlebih dahulu.")