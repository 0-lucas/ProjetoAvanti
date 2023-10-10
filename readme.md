# Classificação de Linguagem de Sinais - ASL

Esse projeto foi desenvolvido com o Atlântico Avanti.  
O objetivo é fazer todo o processo de ponta a ponta de um projeto de visão computacional, que, neste caso, realiza a tradução entre linguagem de sinais e o alfabeto convencional. Para isso, foi realizada a etapa exploratória de pré-processamento da imagem, a modelagem usando Keras e Tensorflow e o deploy final usando Streamlit.  
Tem como base um modelo pré-treinado InceptionV3, e o dataset [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) para treinamento do modelo final.  


## Instalação

Para realizar a instalação, primeiramente clone o repositório e faça a configuração do venv usando o gerenciador de pacotes [pip](https://pip.pypa.io/en/stable/).  
Copie e cole o seguinte código no terminal.  

```bash
pip install requirements.txt
```

Para acesso ao dataset, é utilizado um método interno usando a biblioteca opendownload e uma chave da API do Kaggle.  
O método é encontrado no notebook de pré-processamento.

## Como usar

O deploy foi realizado em Streamlit, podendo ser acessado [clicando aqui](https://classificacaoasl.streamlit.app). No entanto, por questões de performance ou incompatibilidade, se recomenda realizar a execução pelo arquivo *execucao_local.py*.  
Em breve Web App será reformulado, com desempenho e design em mente.

## Contribuição

Solicitações de pull são muito bem-vindas!  
Para qualquer erro, exceções, problemas de desempenho ou sugestão, sinta-se a vontade para entrar em contato abrindo uma issue ou diretamente com os contribuintes.


### Obrigado!!