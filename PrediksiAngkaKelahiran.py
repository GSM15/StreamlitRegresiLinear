import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Baca data
df = pd.read_csv('data.csv')
df.columns = ['Provinsi', 'SP1971', 'SP1980', 'SP1990', 'SP2000', 'SP2010', 'SP2020']

def analyze_province(province_name):
    province_data = df[df['Provinsi'] == province_name].iloc[0]

    years = np.array([1971, 1980, 1990, 2000, 2010, 2020])
    birth_rates = province_data[['SP1971', 'SP1980', 'SP1990', 'SP2000', 'SP2010', 'SP2020']].values

    valid_data = birth_rates != '-'
    years = years[valid_data]
    birth_rates = birth_rates[valid_data].astype(float)

    if len(birth_rates) == 0:
        st.write(f"Tidak ada data valid untuk provinsi {province_name}.")
        return None

    X = years.reshape(-1, 1)
    y = birth_rates.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    future_year = np.array([[2030]])
    future_prediction = model.predict(future_year)[0][0]

    plt.figure(figsize=(12, 6))
    plt.scatter(years, birth_rates, color='blue', label='Data Aktual')
    for i in range(len(years)):
        plt.text(years[i], birth_rates[i], f'{birth_rates[i]:.2f}', fontsize=10, ha='right')
    plt.plot(years, y_pred, color='red', label='Prediksi Regresi Linear')
    plt.scatter(2030, future_prediction, color='green', s=100, label='Prediksi 2030')
    plt.plot([years[-1], 2030], [y_pred[-1][0], future_prediction], color='red', linestyle='--')

    plt.title(f'Analisis Angka Kelahiran di {province_name}')
    plt.xlabel('Tahun')
    plt.ylabel('Angka Kelahiran')
    plt.legend()

    st.pyplot(plt)

    return {
        'province': province_name,
        'mse': mse,
        'rmse': rmse,
        'future_prediction': future_prediction,
        'plot': plt
    }

st.title("Prediksi Angka Kelahiran di Indonesia Berdasarkan Provinsi")
province_name = st.selectbox("Silahkan Pilih Provinsi", ['Cari Provinsi'] + list(df['Provinsi'].unique()))

if province_name and province_name != 'Cari Provinsi':
    result = analyze_province(province_name)
    if result:
        st.text(f"Prediksi 2030 : {result['future_prediction']:.2f}")
        st.text(f"MSE           : {result['mse']:.4f}")
        st.text(f"RMSE          : {result['rmse']:.4f}")
else:
    st.write("Silahkan pilih provinsi di atas untuk melihat Prediksi Angka Kelahiran di Indonesia Berdasarkan Provinsi.")
