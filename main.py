#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

path = Path(__file__).parent

ds = pd.read_csv(path/"credit_data.csv")

# ds.describe()

# ds[ds.income >= 6000000]


# ds.tail() # last 5 rows

# income => renda anual da pessoa
# loan => valor do emprestimo

#%% Retornando valores unicos de cada coluna
np.unique(ds['default'], return_counts=True)


#%%
sns.countplot(x = ds['default'])


# %% Gráfico de histograma olhando pro atributo idade (age)
plt.hist(x = ds['age'])

#%% Gráfico de histograma olhando pro atributo renda anual (income)
plt.hist(x = ds['income'])

#%%#%% Gráfico de histograma olhando pro atributo dívida (loan)
plt.hist(x = ds['loan'])

#%% criando gráfico dinamico a partir da plotly
grafico = px.scatter_matrix(ds, dimensions=['age', 'income','loan'], color= 'default')
grafico.show()

#%% Localizando valores insonsistentes e tratando eles
ds.loc[ds['age'] < 0]

ds2 = ds.drop(ds[ds['age'] < 0].index)
ds2

#%% Usando a média para substituir os valores insonsistentes
ds['age'][ds['age'] >0].mean()

ds.loc[ds['age'] < 0, 'age'] = 40.92
ds.loc[ds['age'] < 0]

ds.head(27)

#%% Tatando valores faltantes
# ds.isnull().sum()


#%%
ds.loc[pd.isnull(ds['age'])]

ds['age'].fillna(ds['age'].mean(), inplace=True)


#%%
ds.loc[pd.isnull(ds['age'])]

#%%
ds.loc[ds['clientid'].isin([29,31,32])]

#%% Divisão entre previsores e classe

X_credit = ds.iloc[:, 1:4].values
type(X_credit)

#%%
y_credit = ds.iloc[:, 4].values

#%%
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

#%% Verificando valores minimos e maximos apos ajuste dos dados para mesma escala

X_credit[:,0].min(),  X_credit[:,1].min(),  X_credit[:,2].min()


#%%
X_credit[:,0].max(),  X_credit[:,1].max(),  X_credit[:,2].max()


#%%  Fazendo a separação dos dados para teste e treinamento

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)

X_credit_treinamento.shape
y_credit_treinamento.shape
#%%
X_credit_teste.shape, y_credit_teste.shape

#%%
import pickle

with open('credit.pkl', mode ='wb') as f:
    pickle.dump([X_credit_treinamento,y_credit_treinamento, X_credit_teste, y_credit_teste], f)

