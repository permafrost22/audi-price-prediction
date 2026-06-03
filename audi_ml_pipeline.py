import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Aula 9/audi.csv')
print(df.info())

contagem_modelos = df['model'].value_counts()
#filtrar apenas model com pelo menos 10 aparições
modelos_validos = contagem_modelos[contagem_modelos >= 10].index
df = df[df['model'].isin(modelos_validos)]
print("=" * 70)

#eda
print("TIPOS DE DADOS")
print(df.dtypes)
print("ESTATÍSTICAS DESCRITIVAS")

# percentil 99 e 1 para possiveis winsorização
print(df.describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]))

print("DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS")
print("\nModelos:\n", df['model'].value_counts())
print("\nTipo de transmissão:\n", df['transmission'].value_counts())
print("\nTipo de combustível:\n", df['fuelType'].value_counts())
print("=" * 70)

#drop first true para evitar colinearidade no getdummies
df_encoded = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'], drop_first=True)
print(f"Shape após encoding: {df_encoded.shape}")
print("=" * 70)

#defininddo colnas numericas e instanciando o scaler, aplicando-o e voltando para df
COLUNAS_NUMERICAS = ['price', 'year', 'tax', 'mpg', 'engineSize']

scaler = MinMaxScaler()
valores_normalizados = scaler.fit_transform(df_encoded[COLUNAS_NUMERICAS])
df_encoded[COLUNAS_NUMERICAS] = valores_normalizados

print("Estatísticas após normalização [0;1]:")
print(df_encoded[COLUNAS_NUMERICAS].describe().round(3))
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Divide em treino (80%) e teste (20%) de forma aleatória mas reproduzível
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTamanho do treino: {X_train.shape[0]} linhas")
print(f"Tamanho do teste:  {X_test.shape[0]} linhas")
print("=" * 70)

#instanciando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

#avaliando o modelo
print("=" * 70)
print("MÉTRICAS DE AVALIAÇÃO (escala normalizada 0-1)")
print("=" * 70)
print(f"  MAE  (Erro Absoluto Médio):         {mae:.4f}")
print(f"  MSE  (Erro Quadrático Médio):        {mse:.4f}")
print(f"  RMSE (Raiz do Erro Quadrático):      {rmse:.4f}")
print(f"  R²   (Coeficiente de Determinação):  {r2:.4f}")
print("=" * 70)

print(y_pred)
print(y_test)

# usando indices do y_test como id e y_pred como previsao
df_sub = pd.DataFrame({'id': y_test.index,'price': y_pred})
df_sub.to_csv('predictions.csv', index=False)
print(df_sub.head(10))

df_sub2 = pd.read_csv('predictions.csv')
print(df_sub2)
print("=" * 70)
    
#criando grafico com matplot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, edgecolors='k', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Previsão perfeita')
plt.xlabel('Preço Real (normalizado)')
plt.ylabel('Preço Previsto (normalizado)')
plt.title('Regressão Linear — Previsão vs. Valor Real')
plt.legend()
plt.tight_layout()
plt.show()
