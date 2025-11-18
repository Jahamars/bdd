
### 1. Isolation Forest
```python
iso_forest = IsolationForest(
    contamination=0.1,  # Ожидаем 10% аномалий
    random_state=42,
    n_estimators=100
)
```
- **Назначение**: Обнаружение аномалий через изоляцию наблюдений
- **Принцип работы**: Строит случайные деревья, аномалии изолируются быстрее
- **Выходные данные**: anomaly_score_iso (оценка аномальности)

### 2. Z-score метод
```python
# Вычисление Z-score для ключевых признаков
for feature in key_features:
    mean = df_ml[feature].mean()
    std = df_ml[feature].std()
    df_ml[f'{feature}_zscore'] = np.abs((df_ml[feature] - mean) / std)
```
- **Назначение**: Статистическое обнаружение выбросов
- **Порог**: |Z| > 3 считается аномалией
- **Признаки**: Transaction_Amount, Transaction_Volume, Frequency_of_Transactions

### 3. PCA (Principal Component Analysis)
```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```
- **Назначение**: Визуализация многомерных данных в 2D пространстве
- **Использование**: Для графического представления кластеров и аномалий

## библиотеки

- **pandas** - обработка и анализ данных
- **numpy** - математические операции
- **scikit-learn** - машинное обучение
- **matplotlib/seaborn** - визуализация
