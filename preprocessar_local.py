import os
import random
from rembg import remove
import cv2
import matplotlib.pyplot as plt

random.seed(0)


def mostra_imagem(i):
    plt.imshow(i)


def deixar_apenas_50_dataset(path):
    for pasta in os.scandir(path):
        imagens = os.listdir(pasta)

        if len(imagens) > 50:
            num_files_to_delete = len(imagens) - 50

            for i in range(num_files_to_delete):
                pass
                imagem_deletar = pasta.path + "/" + random.choice(imagens)
                os.remove(imagem_deletar)
                imagens.remove(os.path.basename(imagem_deletar))


def aplicar_todas_imagens(path):
    def preprocessar_imagem(imagem_cv2):
        img = cv2.imread(imagem_cv2)
        img_no_back = remove(img)
        mostra_imagem(img_no_back)
        img_gray = cv2.cvtColor(img_no_back, cv2.COLOR_RGB2GRAY)

        alpha = 40 / (1.7 * img_gray.mean())
        img_equal = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=8)
        img_equal = cv2.GaussianBlur(img_equal, (5, 5), 0.9, 0.9)
        mostra_imagem(img_equal)
        img_canny = cv2.Canny(img_equal, threshold1=90, threshold2=200)
        img_canny = img_canny / 255

        cv2.imwrite(imagem_cv2, img_canny)

    for pasta in os.scandir(path):
        lista_imagens = os.listdir(pasta)
        for imagem in lista_imagens:
            caminho_imagem = pasta.path + "/" + imagem
            preprocessar_imagem(caminho_imagem)


if __name__ == "__main__":
    caminho = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train/'
    #deixar_apenas_50_dataset(caminho)  CUIDADO AO RODAR
    #aplicar_todas_imagens(caminho)     CUIDADO AO RODAR
