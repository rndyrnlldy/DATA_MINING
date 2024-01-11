import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model
knn = pickle.load(open('KNN.pkl','rb'))

#load dataset
data = pd.read_csv('/content/drive/MyDrive/Data_Mining/Heart Dataset.csv')


st.title('Aplikasi Penyakit Jantung')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Heart Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Heart Disease</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('*Input Dataframe*')
    st.write(data)
    st.write('---')
    st.header('*Profiling Report*')
    st_profile_report(pr)

#train test split
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():

    age = st.sidebar.number_input('Enter your age: ')
    sex  = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    trtbps = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    restecg = st.sidebar.number_input('Resting electrocardiographic results: ')
    thalachh = st.sidebar.number_input('Maximum heart rate achieved: ')
    exng = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    oldpeak = st.sidebar.number_input('oldpeak ')
    slp = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    caa = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thall = st.sidebar.selectbox('thal',(0,1,2))
    
    user_report_data = {
        'Umur':age,
        'sex':sex,
        'Chest Pain Type':cp,
        'Resting Blood Pressure':trtbps,
        'Serum Cholestrol':chol,
        'Fasting Blood Sugar':fbs,
        'resting electrocardiographic':restecg,
        'maximum heart rate achieved':thalachh,
        'exercise induced angina':exng,
        'Previous Peak':oldpeak,
        'Slope':slp,
        'number of major vessels':caa,
        'Thal Rate':thall,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = knn.predict(user_data)
knn_score = accuracy_score(y_test,knn.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena heart'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(knn_score*100)+'%')