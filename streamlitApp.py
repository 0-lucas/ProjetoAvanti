from tensorflow import keras
import streamlit as st
import cv2
import numpy as np

modelo = keras.models.load_model("mymodel.keras")

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T','U',
               'V', 'W', 'X', 'Y', 'Z', 'apagar', 'vazio', 'espaço']
class_mapping = dict(enumerate(class_names))

webcam = cv2.VideoCapture(0)

col1, col2 = st.columns([0.6, 0.4])

col2.subheader("Tabela de conversão alfabética para ASL")
col2.image(cv2.imread("tabela_conversao.jpg"), use_column_width="always")

col1.subheader("Classificação de Linguagem de Sinais - ASL")

frame_placeholder = col1.empty()

while True:
    # Lê um único frame.
    _, frame = webcam.read()

    # Converte as imagens capturadas para RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redimensiona o tamanho do frame de acordo com dados de entrada do modelo.
    image = cv2.resize(image, (200, 200))
    img = np.expand_dims(image, axis=0)

    prediction = modelo.predict(img, verbose=0)

    predicted_class_number = np.argmax(prediction, axis=1).item()
    predicted_class_letter = class_mapping.get(predicted_class_number)
    accuracy_of_prediction = float(np.max(prediction))

    frame_text = f"{predicted_class_letter}, Precisao: {accuracy_of_prediction:.2f}"
    cv2.putText(frame,
                frame_text,
                (300, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255),
                2)

    frame_placeholder.image(frame, channels="RGB")

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
