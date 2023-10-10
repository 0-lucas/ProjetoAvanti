from keras.models import load_model
import streamlit as st
import cv2
from funcoes_locais import video_prediction
from streamlit_webrtc import webrtc_streamer

st.set_page_config(layout="wide")
modelo = load_model("mymodel.keras")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T','U',
               'V', 'W', 'X', 'Y', 'Z', 'apagar', 'vazio', 'espaço']
class_mapping = dict(enumerate(class_names))


col1, col2 = st.columns([0.6, 0.4])

col1.subheader("Classificação de Linguagem de Sinais - ASL")
frame_placeholder = col1.empty()

col2.subheader("Tabela de conversão alfabética para ASL")
col2.image(cv2.imread("tabela_conversao.jpg"), use_column_width="always")


webrtc_streamer(key="example", video_frame_callback=video_prediction)