import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

imagepath = '/home/gpae03/Documentos/Projects/avanti/ProjetoAvanti/asl_alphabet_test/A_test.jpg'
image = cv2.imread(imagepath)

#convertendo para escala de cinza
gray_image =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Plotar imagem original
#redimensionando
width = 200
height = 200
resized_image = cv2.resize(gray_image, (width, height))

clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
clahe_image = clahe.apply(resized_image)
contrasted = cv2.convertScaleAbs(resized_image, alpha=3.5, beta=0)

kernel = np.ones((3,3), np.uint8) 
#enhance_image = cv2.blur(cv2.Canny(equalized_image, threshold1=95, threshold2=100), (3,3), 5)
edges = cv2.Canny(contrasted, threshold1=100, threshold2=250)
sub = cv2.subtract(contrasted,  edges)
clahe_image = clahe.apply(resized_image)
sub = cv2.convertScaleAbs(sub, alpha=1, beta=80)
fusion = cv2.addWeighted(edges, 0.2, sub, 1, 0)



plt.imshow(img_erosion , cmap="gray")
plt.axis('off')
plt.show()
