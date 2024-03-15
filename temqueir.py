# 1- Carregando os dados #
import pandas as pd
# usaremos pd.read para ler o arquivo e csv o tipo dele
arquivo = pd.read_csv('C:/Users/jprib/OneDrive/Área de Trabalho/python_excel/wine_dataset.csv')

#como o arquivo é grande, nao ha necessidade de colocar o .head para mostrar a quantidade de itens
print(arquivo)

#Marcaremos a sessão desejada que no caso é style e usaremos o replace para trocar o RED(tinto) pelo número 0 e o WHITE pelo 1
arquivo['style']= arquivo['style'].replace('red', '0')
arquivo['style']= arquivo['style'].replace('white', '1')

#separando as variaveis
y = arquivo['style']
x = arquivo.drop('style', axis=1)  #removi a coluna style

#CRIANDO TREINO E TESTE
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

#verificar quantas linhas e colunas o 30% vai treinar e testar
print(x_teste.shape)
print(y_teste.shape)

#ARVORE DE CLASSIFICAÇÃO DO SKLEARN
from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

#resultado
resultado = modelo.score(x_teste, y_teste)          # O .SCORE MOSTRA A PRECISAO DO TESTE
print("Probabilidade : ", resultado)


#observações
print('-'*100)

print(y_teste[500:505])
print(x_teste[500:505])