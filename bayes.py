#%%
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

path = Path(__file__).parent.absolute()

ds_risco_credito = pd.read_csv(path/'risco_credito.csv')

X_risco_credito = ds_risco_credito.iloc[:, 0:4].values

y_risco_credito = ds_risco_credito.iloc[:, 4].values

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

# with open('risco_credito.pkl', mode='wb') as f:
#   pickle.dump([X_risco_credito,y_risco_credito], f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito,y_risco_credito)

#Registros que vão ser utilizados para previsão
#historia boa (0), divida alta (0), garantia nenhuma (1), renda > 35 (2) 
#historia ruim (2), divida alta (0), garantia adequada (0), renda < 15 (0)

previsao = naive_risco_credito.predict([ [0,0,1,2], [2,0,0,0] ])

#classes que temos como respostas possível
naive_risco_credito.classes_

#contagem de cada classe resultante após a previsão
naive_risco_credito.class_count_

#probabilidade a priori de cada classe
naive_risco_credito.class_prior_

#=================================================================================================================================#
#Passando para a base de dados de credito feito em outros arquivos

with open(path/'credit.pkl', mode='rb') as f:
  X_credit_treinamento,y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)

# Percentual de acerto do algoritmo comparado com a base de resposta real 93% de acerto
accuracy_score(y_credit_teste, previsoes)

confusion_matrix(y_credit_teste, previsoes)

#=================================================================================================================================#

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))

#=================================================================================================================================#
# Passando para a base de dados do censo feito em outros arquivos
#%%
with open(path/'censo.pkl', mode='rb') as f:
  X_censo_treinamento,y_censo_treinamento, X_censo_teste, y_censo_teste = pickle.load(f)

naive_censo_data = GaussianNB()
naive_censo_data.fit(X_censo_treinamento, y_censo_treinamento)

previsor = naive_censo_data.predict(X_censo_teste)

#Analisando percentual de acerto do algoritmo para a base do Censo
accuracy_score(y_censo_teste, previsor)

confusion_matrix(y_censo_teste, previsor)

cm_censo = ConfusionMatrix(naive_censo_data)
cm_censo.fit(X_censo_treinamento, y_censo_treinamento)
cm_censo.score(X_censo_teste, y_censo_teste)

print(classification_report(y_censo_teste, previsor))