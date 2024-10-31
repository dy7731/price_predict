import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential 
from keras.optimizers import Adam
from streamlit_option_menu import option_menu

df = pd.read_csv('./csv/pepper.csv', index_col='week', encoding="utf-8")

def train_lstm_model(data, target_column):
    
    X = data[['물가지수', '평균기온', '최저기온']]  
    y = data[['소매가']]  # 종속 변수: 가격

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=False)

    scalerX = MinMaxScaler()
    scalerX.fit(X_train)

    X_train_norm = scalerX.transform(X_train)
    X_test_norm = scalerX.transform(X_test)



    scalerY = MinMaxScaler()
    scalerY.fit(y_train)

    y_train_norm = scalerY.transform(y_train)
    y_test_norm = scalerY.transform(y_test)

    
    # LSTM 모델 구성
    model = Sequential()
    model.add(LSTM(32, activation='tanh' ,input_shape=(3, 1)))
    #model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.2))  
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train_norm, y_train_norm,
                    validation_data=(X_test_norm, y_test_norm), epochs=30, batch_size=10, verbose=2)
    
    return model, scalerX, scalerY

# 예측 함수
def make_prediction(model, scalerX, scalerY, input_values):
    input_values = np.array(input_values).reshape(1, -1)
    scaled_input = scalerX.transform(input_values)
    
    
    
    prediction = model.predict(scaled_input)
    predicted_price = scalerY.inverse_transform(prediction)
    
    return predicted_price[0][0]



st.sidebar.page_link('pages/cabbage.py', label='배추')
st.sidebar.page_link('pages/pepper.py', label='고추')
st.sidebar.page_link('pages/onion.py', label='양파')
st.sidebar.page_link('pages/radish.py', label='무')
st.sidebar.page_link('pages/garlic.py', label='마늘')
st.sidebar.markdown('-----------')


with st.sidebar:
    
    물가지수 = st.slider('물가지수를 선택하세요.',1,200)
    평균기온 = st.slider('평균기온을 선택하세요.',-20,40)
    최저기온 =st.slider('최저기온을 선택하세요.',-20,40)

    target_column = 'retail price'
    input_values = [물가지수, 평균기온, 최저기온]

    model, scalerX, scalery = train_lstm_model(df, target_column)

    if st.button('가격 예측하기'):
        predicted_price = make_prediction(model, scalerX, scalery, input_values)
    



pepper = pd.read_csv('./csv/pepper_predicted.csv', encoding="cp949")
pepper['week'] = pd.to_datetime(pepper['week'])

st.title('5대 농산물 가격 예측 프로젝트')
st.text('\n')
st.text('\n')

predicted_price = make_prediction(model, scalerX, scalery, input_values)
st.header(f"예측된 농산물 가격: {predicted_price:.2f}원")

st.text('\n')


st.subheader('고추 소매가 그래프', divider='gray')
st.text('\n')

    
st.line_chart(
        pepper,
        x="week",
        y=["Predicted Values", "Actual Values"],
        color=["#D3D3D3", "#32CD32"])



st.text('\n')
st.header(' Shap value graph')
st.text('\n')

st.image('고추1.png')
st.text('waterfall 그래프')
st.image('고추2.png')
