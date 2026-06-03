# Previsão de Preço de Carros Audi

Modelo de regressão linear para prever preço de Audis com base em
características como ano, motor, transmissão e tipo de combustível.

## Tecnologias
- Python, pandas, scikit-learn, matplotlib

## Como rodar
```bash
pip install -r requirements.txt
python audi_ml_pipeline.py
```

## Etapas do pipeline
1. Filtragem de modelos com menos de 10 registros
2. One-Hot Encoding de variáveis categóricas
3. Normalização MinMax das variáveis numéricas
4. Treino/teste (80/20) com Regressão Linear
5. Avaliação com MAE, RMSE e R²
