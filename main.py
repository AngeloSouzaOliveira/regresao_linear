import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import seaborn as sm; sns.set()
import warnings

plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')


df_Ibov = pd.read_csv('./data/ibov.csv').dropna().iloc[:-1]
df_Ibov = df_Ibov.rename(columns={'Close': 'preco_fechamento'})
ibov_mean = df_Ibov['preco_fechamento'].mean()
ibov_desvio= df_Ibov['preco_fechamento'].std()
ibov_mediana = df_Ibov['preco_fechamento'].median()
ibov_moda = df_Ibov['preco_fechamento'].mode()


print(f'\n-Preço de fechamento Dolar dx')
df_dolar_dx = pd.read_csv('./data/dolar_dx.csv').dropna().iloc[:-1]
df_dolar_dx = df_dolar_dx.rename(columns={'Close': 'preco_fechamento'})
dolar_dx_mean = df_dolar_dx['preco_fechamento'].mean()
print(f'A média dos preços do dolar_dx em 5 anos: {dolar_dx_mean}')
dolar_dx_desvio= df_dolar_dx['preco_fechamento'].std()
print(f'A desvio-padrão dos preços do dolar_dx em 5 anos: {dolar_dx_desvio}')
dolar_dx_mediana = df_dolar_dx['preco_fechamento'].median()
print(f'A mediana dos preços do dolar_dx em 5 anos: {dolar_dx_mediana}')
dolar_dx_moda = df_dolar_dx['preco_fechamento'].mode()
print(f'A moda dos preços do dolar_dx em 5 anos: {dolar_dx_moda}')

print(f'\n-Preço de fechamento Petr4')
df_petr4 = pd.read_csv('./data/petr4.csv').dropna().iloc[:-1]
df_petr4 = df_petr4.rename(columns={'Close': 'preco_fechamento'})
petr4_mean = df_petr4['preco_fechamento'].mean()
print(f'A média dos preços do petr4 em 5 anos: {petr4_mean}')
petr4_desvio= df_petr4['preco_fechamento'].std()
print(f'A desvio-padrão dos preços do petr4 em 5 anos: {petr4_desvio}')
petr4_mediana = df_petr4['preco_fechamento'].median()
print(f'A mediana dos preços do petr4 em 5 anos: {petr4_mediana}')
petr4_moda = df_petr4['preco_fechamento'].mode()
print(f'A moda dos preços do petr4 em 5 anos: {petr4_moda}')

print(f'\n-Palavra Guerra')
df_word_war = pd.read_csv('./data/guerra.csv').dropna().iloc[:-1]
df_word_war = df_word_war.rename(columns={'Guerra': 'data_word'})
word_mean = df_word_war['data_word'].mean()
print(f'A média da palavra guerra em 5 anos: {word_mean}')
word_desvio= df_word_war['data_word'].std()
print(f'A desvio-padrão da palavra guerra em 5 anos: {word_desvio}')
word_mediana = df_word_war['data_word'].median()
print(f'A mediana da palavra guerra em 5 anos: {word_mediana}')
word_moda = df_word_war['data_word'].mode()
print(f'A moda da palavra guerra em 5 anos: {word_moda}')


print(f'\n-Regressão linear simples')

df_Ibov['Date'] = pd.to_datetime(df_Ibov['Date'])
df_dolar_dx['Date'] = pd.to_datetime(df_dolar_dx['Date'])
df_petr4['Date'] = pd.to_datetime(df_petr4['Date'])

begin_date = pd.to_datetime('2018-10-25')
df_ibov_ultimos_5_anos = df_Ibov[df_Ibov['Date'] >= begin_date]
df_dolar_dx_ultimos_5_anos = df_dolar_dx[df_dolar_dx['Date'] >= begin_date]
df_petr4_ultimos_5_anos = df_petr4[df_petr4['Date'] >= begin_date]

X_ibov = df_ibov_ultimos_5_anos['preco_fechamento'].to_numpy()
X_dolar_dx = df_dolar_dx_ultimos_5_anos['preco_fechamento'].to_numpy()
X_petr4 = df_petr4_ultimos_5_anos['preco_fechamento'].to_numpy()

X_series = [X_ibov, X_petr4, X_dolar_dx]
X_matrix = np.vstack(X_series).T

Y_word_war = df_word_war['data_word'].to_numpy()

len(X_ibov)
len(X_dolar_dx)
len(X_petr4)
len(Y_word_war)

coeficientes = []
interceptos = []
y_pred = []

for X in X_series:
    coef_angular, intercepto = np.polyfit(X, Y_word_war, 1)
    coeficientes.append(coef_angular)
    interceptos.append(intercepto)
    y_pred.append(coef_angular * X + intercepto)

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
fig.suptitle('Análise de Regressão Linear Simples: Análise da relação entre os preços de fechamento semanal de ativos financeiros e a incidência mundial no Google Trends para a palavra "Guerra" de 2018 a 2023.')


# Ibov R. Simple
axes[0].scatter(X_series[0], Y_word_war, label="Dados")
axes[0].plot(X_series[0], y_pred[0], color='green', linewidth=2, label='Linha de Regressão')
axes[0].set_xlabel('Variável Independente (X) - preço de fechamento/semanal')
axes[0].set_ylabel('Variável Dependente (Y) - palavra')
axes[0].legend()
axes[0].set_title('Preço semanal do IBOV vs. a palavra guerra')

# Petr4 R. Simple
axes[1].scatter(X_series[1], Y_word_war, label="Dados")
axes[1].plot(X_series[1], y_pred[1], color='black', linewidth=2, label='Linha de Regressão')
axes[1].set_xlabel('Variável Independente (X) - preço de fechamento/semanal')
axes[1].set_ylabel('Variável Dependente (Y) - palavra')
axes[1].legend()
axes[1].set_title('Preço semanal de PETR4 vs. a palavra guerra')

# Dolar DX R. Simple
axes[2].scatter(X_series[2], Y_word_war, label="Dados")
axes[2].plot(X_series[2], y_pred[2], color='blue', linewidth=2, label='Linha de Regressão') 
axes[2].set_xlabel('Variável Independente (X) - preço de fechamento/semanal')
axes[2].set_ylabel('Variável Dependente (Y) - palavra')
axes[2].legend()
axes[2].set_title('Preço semanal de Dollar DX vs. a palavra guerra')

fig.text(0.18, 0.93, 'Investigação da relação entre os preços semanais das ações IBOV, PETR4 e Dollar DX e a frequência da palavra "Guerra" nas pesquisas semanais do Google Trends.', fontsize=12)
fig.text(0.21, 0.91, 'Os dados abrangem um período de 5 anos, de 2018 a 2023. Os gráficos abaixo ilustram essa análise de regressão linear simples.', fontsize=12)
plt.tight_layout()
plt.show()
