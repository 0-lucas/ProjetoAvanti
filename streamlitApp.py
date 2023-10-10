import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from funcoes_locais import video_prediction

st.title("Classificação de Linguagem de Sinais - ASL")
st.sidebar.title("Tabela de Conversão ASL - Alfabético")
st.sidebar.image(cv2.imread("documentos/tabela_conversao.jpg"), use_column_width=True)

webrtc_streamer(
    key="example",
    video_frame_callback=video_prediction,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

