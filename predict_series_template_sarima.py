# Import Libraries
import streamlit as st
import pingouin as pg
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas_datareader import data as pdr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import statistics
import warnings
import time
from datetime import timedelta
warnings.filterwarnings('ignore')

features=[
    "Introduksi",
    "Upload File",
    "Diskripsi",
    "Plot historical inflation",
    "Check stationarity and autocorrelation",
    "Fit SARIMAX model, parameter, prediksi dan evaluasi model",
    ]
menu=pd.DataFrame(features)
#st.write(menu)
#[m,n] =menu.shape
#st.write(m,n)
#st.sidebar.image("logo_stiei.jpg", use_column_width=False)
st.sidebar.markdown('<h3 style="color: White;"> Author: Nasri </h3>', unsafe_allow_html=True)
st.sidebar.markdown('<h5 style="color: White;"> email: nasri@stieimlg.ac.id </h5>', unsafe_allow_html=True)
st.sidebar.markdown('<h1 style="color: White;">Analisis Prediksi (Template) Data Series dengan model SARIMAX</h1>', unsafe_allow_html=True)

model_analisis = st.sidebar.radio('Baca ketentuan penggunaan dengan seksama, Pilih Analisis Prediksi Data Series:', menu)

def intro():
    st.write("## Selamat Datang di Dashboard Analisis Prediksi (Template) Data Series Model SARIMAX.  👋  👋")
    st.write("##### author: m nasri aw, email: nasri@stieimlg.ac.id; lecturer at https://www.stieimlg.ac.id/; Des 2024.")
    st.write(f"##### - Pendekatan analisis: ")
    '''
    1. Aplikasi ini menggunakan bahasa python dengan library utama statsmodels dan streamlit.
    2. Menggunakan model SARIMAX dengan data training dari data series (seasonal, musiman) seperti cuaca, panenan, inflasi, dll.
    3. Diperlukan data dalam bentuk file csv (bisa digunakan contoh file data inflasi yang dapat di download di https://github.com/nasriaw/inflation_indonesia_2006-2024/blob/main/inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv;
    4. File hanya terdiri dari kolom pertama kolom waktu yang bersifat series bisa dalam bentuk format date time seperti: jam, hari, minggu, bulanan, tahunan atau musiman yang berulang secata time series, kolom ke dua data series misal data cuaca, harga, perjalanan, inflasi atau sejenis nya, dll.
    5. Untuk menggunakan template ini langkah ke-2 Upload File, silahkan menggunakan data series dengan format seperti diatas no,3 dan no.4 dan diupload.    
    6. Analisis Prediksi Data Series meliputi:
       1. Diskripsi.
       2. Plot historical inflation.
       3. Check stationarity and autocorrelation.
       4. Fit SARIMAX model: parameter, prediksi dan evaluasi,
    7. Fitting model (langkah 6.4), untuk menentukan parameter  model SARIMAX yang optimal, menggunakan auto_arima, memerlukan waktu sesuai dengan pola data musiman dan besarnya data yang diberikam, untuk contoh data inflasi memerlukan waktu sekitar 4-5 menit. 
       ###### 👈 Pilih Menu di sebelah; Pastikan data telah di upload (langkah ke-2: Upload File)
    8. Untuk link demo silahkan klik https://nasriaw-aw-predict.streamlit.app/ atau di https://huggingface.co/spaces/nasriaw/predict_inflation.
       Selamat belajar semoga memudahkan untuk memahami Analisis Prediksi Data Series.
    '''
    return intro

def open_file():
    if 'data' not in st.session_state:
        st.session_state.data = None

    def load_data():
        st.session_state.data = pd.read_csv(st.session_state.loader)

    file = st.file_uploader('Choose a file', type='csv', key='loader', on_change=load_data)

    df = st.session_state.data
    if df is not None:
        # Run program
        st.write('Gunakan Browse Files jika upload data baru.')
    return df
#df=open_file()

def descriptive():
    df=open_file()
    df= df.dropna()
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0])
    st.write("### 1. Data Head dan Statistik Diskripsi ") #{df.iloc[:,1]}")
    st.write(f"#### dimensi data: {df.shape}")
    st.write("#### Data Head : ")
    st.write(df.head())
    st.write("#### Data Diskripsi Inflasi : ")
    st.write(df.iloc[:,1].describe())#(include='all').fillna("").astype("str"))
    
def historical():
    df=open_file()
    df.set_index(df.iloc[:,0], inplace=True)
    st.write("### Inflation Data (2006-2024) Line Chart")
    #st.line_chart(df.iloc[:,1]) #scatter, bar, line, area, altair
    st.line_chart(df.iloc[:,1], x_label="bulan", y_label="inflation, %") #scatter, bar, line, area, altair
    
def check_stationarity():
    # S3: Check stationarity and autocorrelation
    # Plot ACF and PACF to identify ARIMA parameters
    df=open_file()
    st.write("### Auto Correlation Function (ACF) Chart")
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))
    plot_acf(df.iloc[:,1], ax=ax1)
    st.pyplot(fig)
    st.write("### Partial Auto Correlation Function (PCF) Chart")
    #st.line_chart(df.iloc[:,1], x_label="periode", y_label="inflasi")
    
    fig1, (ax2) = plt.subplots(1, 1, figsize=(8, 3))
    plot_pacf(df.iloc[:,1], ax=ax2)
    st.pyplot(fig1)
    
    st.write("### Augmented Dickey-Fuller (ADF), P-value Result")
    result = adfuller(df.iloc[:,1])
    st.write("Augmented Dickey-Fuller (ADF) Statistic:", result[0].round(4))
    st.write("p-value:", result[1].round(4))
       
    for key, value in result[4].items():
        st.write(f'Critical Value ({key}): {value.round(4)}')
 
    #st.write(f'result[0] = {result[0]:.4f}')
    #st.write(f'result[4] = {result[4]["5%"]:.4f}')
    st.write('### Kesimpulan :')
    if result[0] < result[4]["5%"]:
        st.write(f'ADF result = {result[0]:.4f} < {result[4]["5%"]:.4f} (Critical value=5%) : Hipotesis stasioner ditolak, Pola data tidak stasioner, yang berarti menunjukkan tren atau musiman dan yang tidak memiliki rata-rata dan varians yang konstan dari waktu ke waktu.')
    else:
        st.write(f'ADF result = {result[0]:.4f} > {result[4]["5%"]:.4f} (Critical value=5%) : Hipotesis stasioner diterima, Pola data stasioner, yang berarti menunjukkan tren atau musiman dan yang memiliki rata-rata dan varians yang konstan dari waktu ke waktu.')
    
    if result[1].round(4)> 0.05:
        st.write(f'p-value = {result[1]:.4f} > 0.05 : Hipotesis stasioner diterima, Pola data stasioner, yang berarti menunjukkan tren atau musiman dan yang memiliki rata-rata dan varians yang konstan dari waktu ke waktu, parameter enforce_stationarity=False.')
    else:
        st.write(f'p-value = {result[1]:.4f} < 0.05 : Hipotesis stasioner ditolak, Pola data tidak stasioner, yang berarti menunjukkan tren atau musiman dan yang tidak memiliki rata-rata dan varians yang konstan dari waktu ke waktu, parameter enforce_stationarity=True.')


def SARIMAX_model():
    df=open_file()
    df.set_index(df.iloc[:,0], inplace=True)
    
    start_time = time.time()
    with st.spinner("Tunggu proses optimasi parameter SARIMAX, untuk data inflasi, waktu sekitar 5 menit. Sebaiknya di jalankan secara offline, source code di download dan menjalankan perintah di prompt terminal: $streamlit run app.py ", show_time=True):
        model = auto_arima(df.iloc[:,1], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(df.iloc[:,1])
        end_time = time.time()
        time.sleep=end_time
        time_lapsed =np.mean(end_time - start_time)
        st.success(f"Selesai !!, waktu optimasi parameter SARIMAX : {str(timedelta(seconds=time_lapsed))} detik': ")
           
    st.write("Optimal parameter : ")
    st.write(model)
    st.write(f'parameter order optimal, p,d,q : {model.order}')
    st.write(f'parameter seasonal order optimal P,D,Q,m : {model.seasonal_order}')
    
    st.write("Output Optimized model SARIMAX : ") 
    optimized_model = SARIMAX(
        df.iloc[:,1],  
        order=model.order[:3],  						# Non-seasonal parameters
        seasonal_order=model.seasonal_order[:4],  	# Seasonal parameters
        enforce_stationarity=True, # False if p>0.05,
        enforce_invertibility=False,
    )
 
    optimized_sarima_fit = optimized_model.fit(disp=False)
    st.write("### SARIMAX RESULT")
    st.write(optimized_sarima_fit.summary())

    st.write('### Standardized Residuals')
    fig=optimized_sarima_fit.plot_diagnostics(figsize=(12, 8))
    st.pyplot(fig)
    '''
    Residual tampak acak dan berfluktuasi di sekitar nol, yang menunjukkan tidak ada pola atau tren yang terlihat. Histogram dengan Kepadatan Diperkirakan, residual terdistribusi secara normal, karena histogram selaras dengan kurva kepadatan normal. Plot Q-Q Normal, residual sebagian besar mengikuti garis diagonal merah, yang memvalidasi bahwa residual tersebut hampir terdistribusi secara normal. Korelogram (ACF), tidak ada lonjakan signifikan dalam fungsi autokorelasi (ACF), yang menunjukkan residual tidak berkorelasi.
    '''
    
    train = df.iloc[:-24] 
    test = df.iloc[-24:]
    train.set_index(train.iloc[:,0], inplace=True)
    test.set_index(test.iloc[:,0], inplace=True)
    
    forecast_test_optimized = optimized_sarima_fit.forecast(steps=24)
    forecast_test_optimized.index = test.index
    
    st.write("### Data Train Line Chart")
    st.line_chart(train.iloc[:,1], x_label="bulan, train", y_label="inflasi, %")
    st.write("### Test Line Chart")
    st.line_chart(test.iloc[:,1],x_label="bulan, testing", y_label="inflasi testing, %")
    st.write("### Forecast Line Chart")
    st.line_chart(forecast_test_optimized, x_label="bulan forecast", y_label="inflasi Forecast, %")
    
    predicted_values = forecast_test_optimized.values 
    actual_values = test.iloc[:,1] #values #.flatten()

    prediction_df = pd.DataFrame({
        'actual': actual_values.round(4),
        'predicted': predicted_values.round(4),
        'deviation': predicted_values-actual_values.round(4),
        'deviation^2': (predicted_values-actual_values.round(4))**2
    })
    st.write("### Actual - Predicted - Deviation")
    st.write(prediction_df)
    
    data = {
        "Actual Inflation": actual_values,
        "Predicted Inflation": predicted_values,}
    df = pd.DataFrame(data) #, index=dates)
    column_names=list(df)
    #column_names[0]
    #column_names[1]
    
    st.write("### Actual - Predicted Line Chart")
    st.line_chart(df, x_label="bulan", y_label="inflation, %") #.iloc[:,1], x_label=column_names[0], y_label=column_names[1])

    # Evaluation Model
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_values - actual_values))
    st.write("### Evaluation Indicator")
    st.write("MAE:", mae.round(4))

    # Root Mean Squared Error (RMSE)
    mse = np.mean((predicted_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse.round(4))
    st.write("MSE:", mse.round(4))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((predicted_values - actual_values) / actual_values)) * 100
    st.write("MAPE:", mape.round(4))
    
    # Standard Deviation Predicted
    std=np.mean(statistics.stdev(predicted_values))
    st.write(f'Standard Deviation : {(std.round(4))}')

    # Forecast for the next 3 month
    st.write("### Forecast next 3 month")
    forecast_steps = 3
    forecast, stderr, conf_int = optimized_sarima_fit.forecast(steps=forecast_steps)

    # Convert forecast to a pandas Series for easier plotting
    forecast_series = pd.Series(forecast, index=pd.date_range('2025', periods=forecast_steps, freq='ME'))
    st.write(f'Inflation Predicted next 3 month: {forecast:3f} %') #, stderr: {stderr}, conf: {conf_int}')
      
if model_analisis == "Introduksi":
    intro()
elif model_analisis == "Upload File":
    open_file()
elif model_analisis == "Diskripsi":
    descriptive()
elif model_analisis == "Plot historical inflation":
    historical()
elif model_analisis == "Check stationarity and autocorrelation":
    check_stationarity()
else:
    SARIMAX_model()
