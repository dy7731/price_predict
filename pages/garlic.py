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
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(layout="wide")

df = pd.read_csv('./csv/garlic_week.csv', index_col='week', encoding="cp949")

st.sidebar.page_link('pages/cabbage.py', label='배추', icon='🥬')
st.sidebar.page_link('pages/onion.py', label='양파', icon='🧅')
st.sidebar.page_link('pages/garlic.py', label='마늘', icon='🤍')
st.sidebar.page_link('pages/radish.py', label='무', icon='🥔')
st.sidebar.page_link('pages/pepper.py', label='고추', icon='🌶️')

@st.cache_resource
def train_lstm_model(data, target_column):
    
    X = data[['환율','생산량', '재배면적']]  
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
    
    return model, scalerX, scalerY, X_test_norm, y_test_norm

# 예측 함수
def make_prediction(model, scalerX, scalerY, input_values):
    # 입력값을 데이터프레임으로 변환하여 feature names를 유지
    input_df = pd.DataFrame([input_values], columns=['환율','생산량', '재배면적'])
    
    # 스케일링
    scaled_input = scalerX.transform(input_df)
    # 예측
    prediction = model.predict(scaled_input)
    predicted_price = scalerY.inverse_transform(prediction)
    
    return predicted_price[0][0]

# 성능 평가 함수
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 퍼센트로 변환
    return mae, rmse, mape




with st.sidebar:
    
    환율 = st.slider('환율(원)을 선택하세요.',800,1500)
    생산량 = st.slider('생산량을 선택하세요. (톤)',float(df['생산량'].min()),float(df['생산량'].max()))
    재배면적 = st.slider('재배면적을 선택하세요. (헥타르)',float(df['재배면적'].min()),float(df['재배면적'].max()))
    
    

    target_column = 'retail price'
    input_values = [환율,생산량, 재배면적]

    model, scalerX, scalerY, X_test_norm, y_test_norm = train_lstm_model(df, target_column)

    if st.button('가격 예측하기'):
        predicted_price = make_prediction(model, scalerX, scalerY, input_values)
        
    



garlic = pd.read_csv('./csv/garlic_predicted.csv', encoding="cp949")

garlic['week'] = pd.to_datetime(garlic['week'])

st.title('5대 농산물 주간가격 예측 프로젝트')
st.markdown('-----------')

st.text('\n')

predicted_price = make_prediction(model, scalerX, scalerY, input_values)
st.header(f"예측된 마늘의 가격: {predicted_price:.2f}원")


st.text('\n')


st.subheader('마늘 소매가 그래프', divider='gray')
st.text('\n')

    
st.line_chart(
        garlic,
        x="week",
        y=["Predicted Values", "actual Values"],
        color=["#D3D3D3", "#32CD32"])



# 성능 평가 및 결과 출력
y_test_inverse = scalerY.inverse_transform(y_test_norm)  # 정규화된 y_test 값을 역변환
y_pred_inverse = model.predict(X_test_norm)  # 정규화된 예측값
y_pred_inverse = scalerY.inverse_transform(y_pred_inverse)  # 예측값 역변환

mae, rmse, mape = calculate_metrics(y_test_inverse, y_pred_inverse)

metrics_data = {
    'Metric': ['MAE', 'RMSE', 'MAPE'],
    'Value': [mae, rmse, mape]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.set_index('Metric', inplace=True)  # 'Metric' 컬럼을 인덱스로 설정
metrics_df.index.name = 'Metrics'  # 인덱스 이름을 'Metrics'로 설정

st.subheader('모델 성능 평가')
st.table(metrics_df)



