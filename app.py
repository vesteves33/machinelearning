#%%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

path = Path(__file__).parent.absolute()
ds_censo = pd.read_csv(path/"census.csv" )

ds_censo.describe()

#%% Analisando se há valores faltantes
ds_censo.isnull().sum()

#%%
np.unique(ds_censo['income'], return_counts=True)

sns.countplot(x= 'income', data=ds_censo)

#%% Gerando alguns histogramas para verificar a distribuição dos dados

plt.hist(x = ds_censo['age'])

#%% Gerando alguns histogramas para verificar a distribuição dos dados
plt.hist(x = ds_censo['education-num'])


#%%
plt.hist(x = ds_censo['hour-per-week'])

#%% 
grafico = px.treemap(ds_censo, path=['workclass','age'])
grafico.show()

#%% 
grafico = px.treemap(ds_censo, path=['occupation','relationship','age'])
grafico.show()

#%% 
grafico = px.parallel_categories(ds_censo, dimensions=['occupation','relationship'])
grafico.show()

#%% 
grafico = px.parallel_categories(ds_censo, dimensions=['workclass','occupation','income'])
grafico.show()

#%% 
grafico = px.parallel_categories(ds_censo, dimensions=['education','income'])
grafico.show()

#%% Divisão entre previsores e classe
X_censo = ds_censo.iloc[:,0:14].values

#%% Divisão entre previsores e classe
y_censo = ds_censo.iloc[:,14].values

#%% Usando o Label Encoder para transformar os dados

label_enconder_teste = LabelEncoder()

teste = label_enconder_teste.fit_transform(X_censo[:,1])

#%% Criando os LabelEncoder de cada coluna
label_enconder_workclass = LabelEncoder()
label_enconder_education = LabelEncoder()
label_enconder_marital = LabelEncoder()
label_enconder_occupation = LabelEncoder()
label_enconder_relationship = LabelEncoder()
label_enconder_race = LabelEncoder()
label_enconder_sex = LabelEncoder()
label_enconder_country = LabelEncoder()

#%%
X_censo[:,1] = label_enconder_workclass.fit_transform(X_censo[:,1])
X_censo[:,3] = label_enconder_education.fit_transform(X_censo[:,3])
X_censo[:,5] = label_enconder_marital.fit_transform(X_censo[:,5])
X_censo[:,6] = label_enconder_occupation.fit_transform(X_censo[:,6])
X_censo[:,7] = label_enconder_relationship.fit_transform(X_censo[:,7])
X_censo[:,8] = label_enconder_race.fit_transform(X_censo[:,8])
X_censo[:,9] = label_enconder_sex.fit_transform(X_censo[:,9])
X_censo[:,13] = label_enconder_country.fit_transform(X_censo[:,13])

X_censo[0]
X_censo


#%% Fazendo a trransformação dos dados antes do processamento pelo futuro algoritmo

onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),[1,3,5,6,7,8,9,13])], remainder='passthrough')

X_censo = onehotencoder_censo.fit_transform(X_censo).toarray()


#%% Fazendo padronização da base de dados com StandardScaler

scaler_censo = StandardScaler()
X_censo = scaler_censo.fit_transform(X_censo)

X_censo[0]


#%% 
from sklearn.model_selection import train_test_split
X_censo_treinamento, X_censo_teste, y_censo_treinamento, y_censo_teste = train_test_split(X_censo, y_censo, test_size=0.15, random_state=0)

X_censo_treinamento.shape, y_censo_treinamento.shape


#%%
X_censo_teste.shape, y_censo_teste.shape


#%% Usando biblioteca Pickle para salvar as bases de dados separadas no passo a passo das aulas, devidamente ajustadas, para que possam ser usadas posteriormente 

import pickle

with open('censo.pkl', mode ='wb') as f:
    pickle.dump([X_censo_treinamento,y_censo_treinamento,X_censo_teste,y_censo_teste],f)
