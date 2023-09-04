import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

imagepath = '/home/gpae03/Documentos/Projects/avanti/ProjetoAvanti/asl_alphabet_test/B_test.jpg'
image = cv2.imread(imagepath)

#convertendo para escala de cinza
gray_image =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Plotar imagem original
#redimensionando
width = 200
height = 200
resized_image = cv2.resize(gray_image, (width, height))

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
clahe_image = clahe.apply(resized_image)
contrasted = cv2.convertScaleAbs(resized_image, alpha=3.5, beta=0)
sub = cv2.subtract(contrasted,  -gray_image)

ret,thresh = cv2.threshold(sub,20, 50,cv2.THRESH_BINARY)




# Defina as coordenadas do canto superior esquerdo (x, y) e as dimensões (largura e altura) da região a ser recortada
x = 25  # coordenada x do canto superior esquerdo
y = 25   # coordenada y do canto superior esquerdo
largura = 200  # largura da região
altura = 200   # altura da região

# Recorte a região de interesse (ROI) da imagem original
regiao_recortada = thresh[y:y+altura, x:x+largura]

# Exiba a região recortada
plt.imshow(regiao_recortada , cmap="gray")
plt.axis('off')
plt.show()