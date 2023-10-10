import cv2
import numpy as np
from keras.models import load_model

model = load_model("modelo_treinado.keras")


class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', 'apagar', 'vazio', 'espaço']

class_mapping = dict(enumerate(class_names))

video = cv2.VideoCapture(0)

while True:
    # Lê um único frame.
    _, frame = video.read()

    # Converte o frame lido para RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redimensiona o tamanho do frame de acordo com dados de entrada do modelo.
    image = cv2.resize(image, (200, 200))
    img = np.expand_dims(image, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class_number = int(np.argmax(prediction, axis=1))
    predicted_class_letter = class_mapping.get(predicted_class_number)
    accuracy_of_prediction = float(np.max(prediction))

    frame_text = f"{predicted_class_letter}, Precisao: {accuracy_of_prediction:.2f}"
    cv2.putText(frame,
                frame_text,
                (300, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255),
                2)

    cv2.imshow("Classificacao ASL", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break