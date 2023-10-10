import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from funcoes_locais import video_prediction

st.title("Classificação de Linguagem de Sinais - ASL")
st.sidebar.title("Tabela de Conversão ASL - Alfabético")
st.sidebar.image(cv2.imread("tabela_conversao.jpg"), use_column_width=True)

# class VideoProcessor:
#     def recv(self, capture):
#         class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
#                        'T',
#                        'U', 'V', 'W', 'X', 'Y', 'Z', 'apagar', 'vazio', 'espaço']
#
#         class_mapping = dict(enumerate(class_names))
#
#         prediction_model = load_model("mymodel.keras")
#
#         # Converte o frame lido para RGB.
#
#         image = frame = capture.to_ndarray(format="bgr24")
#
#         # Redimensiona o tamanho do frame de acordo com dados de entrada do modelo.
#         image = cv2.resize(image, (200, 200))
#         image = np.expand_dims(image, axis=0)
#
#         prediction = prediction_model.predict(image, verbose=0)
#
#         predicted_class_number = int(np.argmax(prediction, axis=1))
#         predicted_class_letter = class_mapping.get(predicted_class_number)
#         accuracy_of_prediction = float(np.max(prediction))
#
#         frame_text = f"{predicted_class_letter}, Precisao: {accuracy_of_prediction:.2f}"
#         cv2.putText(frame,
#                     frame_text,
#                     (300, 450),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (255, 255, 255),
#                     2)
#
#         return av.VideoFrame.from_ndarray(frame, format="bgr24")
#

webrtc_streamer(
    key="example",
    video_frame_callback=video_prediction,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

