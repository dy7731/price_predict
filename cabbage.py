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

df = pd.read_csv(('./data/cabbage_week.csv'), index_col='week', encoding="cp949")


@st.cache_resource
def train_lstm_model(data, target_column):
    
    X = data[['수출액','수출량', '평균기온', '최저기온']]  
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
    model.add(LSTM(32, activation='tanh' ,input_shape=(4, 1)))
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
    # 입력값을 데이터프레임으로 변환하여 feature names를 유지
    input_df = pd.DataFrame([input_values], columns=['수출액', '수출량', '평균기온', '최저기온'])
    
    # 스케일링
    scaled_input = scalerX.transform(input_df)
    
    
    
    # 예측
    prediction = model.predict(scaled_input)
    predicted_price = scalerY.inverse_transform(prediction)
    
    return predicted_price[0][0]







with st.sidebar:
    choice = option_menu('Menu',['배추','양파','고추','무','마늘'],
                          menu_icon="cast")
    수출액 = st.slider('수출금액를 선택하세요. (만원)',float(df['수출액'].min()),float(df['수출액'].max()))
    수출량 = st.slider('수출량을 선택하세요. (kg)',float(df['수출량'].min()),float(df['수출량'].max()))
    평균기온 = st.slider('평균기온을 선택하세요.',-20,40)
    최저기온 =st.slider('최저기온을 선택하세요.',-20,40)

    target_column = 'retail price'
    input_values = [수출액, 수출량, 평균기온, 최저기온]

    model, scalerX, scalery = train_lstm_model(df, target_column)

    if st.button('가격 예측하기'):
        predicted_price = make_prediction(model, scalerX, scalery, input_values)
        
    



pepper = pd.read_csv('./data/cabbage_predicted.csv', encoding="cp949")

pepper['week'] = pd.to_datetime(pepper['week'])

st.title('5대 농산물 주간가격 예측 프로젝트')
st.text('\n')
st.text('\n')

predicted_price = make_prediction(model, scalerX, scalery, input_values)
st.header(f"예측된 배추의 가격: {predicted_price:.2f}원")


st.text('\n')


st.subheader('배추 소매가 그래프', divider='gray')
st.text('\n')

    
st.line_chart(
        pepper,
        x="week",
        y=["Predicted Values", "actual Values"],
        color=["#FF0000", "#0000FF"])



st.text('\n')
st.header(' Shap value graph')
st.text('\n')

st.image('배추1.png')
st.text('waterfall 그래프')
st.image('배추2.png')
    




