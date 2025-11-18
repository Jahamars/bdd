import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data.csv')

print("=" * 80)
print("АНАЛИЗ ДАННЫХ О ТРАНЗАКЦИЯХ С ИСПОЛЬЗОВАНИЕМ МАШИННОГО ОБУЧЕНИЯ")
print("=" * 80)
print("\n1. ОБЗОР ДАННЫХ")
print("-" * 80)
print(df.head())
print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nПропущенные значения:\n{df.isnull().sum()}")

# Подготовка данных для ML
print("\n" + "=" * 80)
print("2. ПОДГОТОВКА ДАННЫХ ДЛЯ МАШИННОГО ОБУЧЕНИЯ")
print("-" * 80)

# Создаем копию для обработки
df_ml = df.copy()

# Кодируем категориальные переменные
le_gender = LabelEncoder()
le_account = LabelEncoder()
le_day = LabelEncoder()
le_time = LabelEncoder()

df_ml['Gender_encoded'] = le_gender.fit_transform(df_ml['Gender'])
df_ml['Account_Type_encoded'] = le_account.fit_transform(df_ml['Account_Type'])
df_ml['Day_of_Week_encoded'] = le_day.fit_transform(df_ml['Day_of_Week'])
df_ml['Time_of_Day_encoded'] = le_time.fit_transform(df_ml['Time_of_Day'])

# Выбираем признаки для анализа
features = ['Transaction_Amount', 'Transaction_Volume', 'Average_Transaction_Amount',
            'Frequency_of_Transactions', 'Time_Since_Last_Transaction',
            'Age', 'Income', 'Gender_encoded', 'Account_Type_encoded',
            'Day_of_Week_encoded', 'Time_of_Day_encoded']

X = df_ml[features]

print(f"Используемые признаки: {len(features)}")
print(f"Признаки: {', '.join(features[:7])}...")

# Стандартизация данных (важно для ML!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✓ Данные стандартизированы (mean=0, std=1)")

# МЕТОД 1: Isolation Forest
print("\n" + "=" * 80)
print("3. ОБНАРУЖЕНИЕ АНОМАЛИЙ: ISOLATION FOREST")
print("-" * 80)
print("Isolation Forest - это алгоритм, который изолирует аномалии,")
print("строя случайные деревья решений. Аномалии легче изолировать,")
print("так как они находятся 'далеко' от основной массы данных.")

# Обучаем модель
iso_forest = IsolationForest(
    contamination=0.1,  # Ожидаем 10% аномалий
    random_state=42,
    n_estimators=100
)

# Предсказываем аномалии (-1 = аномалия, 1 = нормальная транзакция)
df_ml['anomaly_iso'] = iso_forest.fit_predict(X_scaled)
df_ml['anomaly_score_iso'] = iso_forest.score_samples(X_scaled)

# Статистика
n_anomalies_iso = (df_ml['anomaly_iso'] == -1).sum()
print(f"\n✓ Обнаружено аномалий: {n_anomalies_iso} из {len(df_ml)} ({n_anomalies_iso/len(df_ml)*100:.1f}%)")

print("\nАНОМАЛЬНЫЕ ТРАНЗАКЦИИ (Isolation Forest):")
print("-" * 80)
anomalies_iso = df_ml[df_ml['anomaly_iso'] == -1][['Transaction_ID', 'Transaction_Amount', 
                                                      'Transaction_Volume', 'Frequency_of_Transactions',
                                                      'Income', 'Age', 'anomaly_score_iso']]
print(anomalies_iso.to_string(index=False))

# МЕТОД 2: Статистический подход (Z-score)
print("\n" + "=" * 80)
print("4. ОБНАРУЖЕНИЕ АНОМАЛИЙ: Z-SCORE (СТАТИСТИЧЕСКИЙ МЕТОД)")
print("-" * 80)
print("Z-score показывает, насколько значение отклоняется от среднего")
print("в единицах стандартного отклонения. |Z| > 3 считается аномалией.")

# Вычисляем Z-score для ключевых признаков
key_features = ['Transaction_Amount', 'Transaction_Volume', 'Frequency_of_Transactions']
for feature in key_features:
    mean = df_ml[feature].mean()
    std = df_ml[feature].std()
    df_ml[f'{feature}_zscore'] = np.abs((df_ml[feature] - mean) / std)

# Транзакция аномальна, если хотя бы один Z-score > 3
df_ml['is_anomaly_zscore'] = df_ml[[f'{f}_zscore' for f in key_features]].max(axis=1) > 3

n_anomalies_zscore = df_ml['is_anomaly_zscore'].sum()
print(f"\n✓ Обнаружено аномалий: {n_anomalies_zscore} из {len(df_ml)} ({n_anomalies_zscore/len(df_ml)*100:.1f}%)")

if n_anomalies_zscore > 0:
    print("\nАНОМАЛЬНЫЕ ТРАНЗАКЦИИ (Z-score):")
    print("-" * 80)
    anomalies_z = df_ml[df_ml['is_anomaly_zscore']][['Transaction_ID', 'Transaction_Amount',
                                                       'Transaction_Amount_zscore',
                                                       'Transaction_Volume', 'Transaction_Volume_zscore']]
    print(anomalies_z.to_string(index=False))
else:
    print("\nАномалий по Z-score не обнаружено (все значения в пределах 3σ)")

# Анализ признаков
print("\n" + "=" * 80)
print("5. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
print("-" * 80)
print("\nСТАТИСТИКА ПО КЛЮЧЕВЫМ ПРИЗНАКАМ:")
stats = df_ml[['Transaction_Amount', 'Transaction_Volume', 'Frequency_of_Transactions', 
               'Time_Since_Last_Transaction', 'Age', 'Income']].describe()
print(stats)

# Сравнение аномальных и нормальных транзакций
print("\n" + "=" * 80)
print("6. СРАВНЕНИЕ АНОМАЛЬНЫХ И НОРМАЛЬНЫХ ТРАНЗАКЦИЙ")
print("-" * 80)

normal = df_ml[df_ml['anomaly_iso'] == 1]
anomaly = df_ml[df_ml['anomaly_iso'] == -1]

comparison = pd.DataFrame({
    'Признак': ['Transaction_Amount', 'Transaction_Volume', 'Frequency_of_Transactions', 'Income', 'Age'],
    'Норма (среднее)': [normal['Transaction_Amount'].mean(), 
                        normal['Transaction_Volume'].mean(),
                        normal['Frequency_of_Transactions'].mean(),
                        normal['Income'].mean(),
                        normal['Age'].mean()],
    'Аномалия (среднее)': [anomaly['Transaction_Amount'].mean() if len(anomaly) > 0 else 0,
                           anomaly['Transaction_Volume'].mean() if len(anomaly) > 0 else 0,
                           anomaly['Frequency_of_Transactions'].mean() if len(anomaly) > 0 else 0,
                           anomaly['Income'].mean() if len(anomaly) > 0 else 0,
                           anomaly['Age'].mean() if len(anomaly) > 0 else 0]
})
print(comparison.to_string(index=False))

# Визуализация
print("\n" + "=" * 80)
print("7. СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Обнаружение аномалий в транзакциях с помощью ML', fontsize=16, fontweight='bold')

# График 1: Transaction Amount vs Volume
ax1 = axes[0, 0]
normal_data = df_ml[df_ml['anomaly_iso'] == 1]
anomaly_data = df_ml[df_ml['anomaly_iso'] == -1]

ax1.scatter(normal_data['Transaction_Amount'], normal_data['Transaction_Volume'], 
           c='blue', label='Нормальные', alpha=0.6, s=100)
ax1.scatter(anomaly_data['Transaction_Amount'], anomaly_data['Transaction_Volume'], 
           c='red', label='Аномалии', alpha=0.8, s=150, marker='X')
ax1.set_xlabel('Сумма транзакции', fontsize=11)
ax1.set_ylabel('Объем транзакции', fontsize=11)
ax1.set_title('Сумма vs Объем транзакции', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Anomaly Score Distribution
ax2 = axes[0, 1]
ax2.hist(df_ml['anomaly_score_iso'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(df_ml[df_ml['anomaly_iso'] == -1]['anomaly_score_iso'].max(), 
           color='red', linestyle='--', linewidth=2, label='Порог аномалии')
ax2.set_xlabel('Anomaly Score', fontsize=11)
ax2.set_ylabel('Частота', fontsize=11)
ax2.set_title('Распределение Anomaly Score', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Frequency vs Income
ax3 = axes[1, 0]
ax3.scatter(normal_data['Income'], normal_data['Frequency_of_Transactions'], 
           c='blue', label='Нормальные', alpha=0.6, s=100)
ax3.scatter(anomaly_data['Income'], anomaly_data['Frequency_of_Transactions'], 
           c='red', label='Аномалии', alpha=0.8, s=150, marker='X')
ax3.set_xlabel('Доход', fontsize=11)
ax3.set_ylabel('Частота транзакций', fontsize=11)
ax3.set_title('Доход vs Частота транзакций', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# График 4: PCA визуализация
ax4 = axes[1, 1]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_ml['PCA1'] = X_pca[:, 0]
df_ml['PCA2'] = X_pca[:, 1]

normal_pca = df_ml[df_ml['anomaly_iso'] == 1]
anomaly_pca = df_ml[df_ml['anomaly_iso'] == -1]

ax4.scatter(normal_pca['PCA1'], normal_pca['PCA2'], 
           c='blue', label='Нормальные', alpha=0.6, s=100)
ax4.scatter(anomaly_pca['PCA1'], anomaly_pca['PCA2'], 
           c='red', label='Аномалии', alpha=0.8, s=150, marker='X')
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)', fontsize=11)
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)', fontsize=11)
ax4.set_title('PCA: 2D визуализация данных', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("✓ Графики сохранены в 'anomaly_detection_results.png'")

# Сохранение результатов
print("\n" + "=" * 80)
print("8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 80)

output_df = df.copy()
output_df['Is_Anomaly'] = df_ml['anomaly_iso'].map({1: 'Normal', -1: 'Anomaly'})
output_df['Anomaly_Score'] = df_ml['anomaly_score_iso']
output_df.to_csv('transactions_with_anomalies.csv', index=False)
print("✓ Результаты сохранены в 'transactions_with_anomalies.csv'")

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80)
print("""
1. Isolation Forest обнаружил аномальные транзакции на основе множества признаков
2. Аномалии - это транзакции, которые значительно отличаются от типичного поведения
3. Для детекции использовались: сумма, объем, частота, доход, возраст и др.
4. Такой подход может использоваться для:
   - Обнаружения мошенничества
   - Выявления ошибок в данных
   - Мониторинга необычного поведения клиентов
   
РЕКОМЕНДАЦИИ:
- Проверьте аномальные транзакции вручную
- Настройте порог contamination в зависимости от ваших данных
- Используйте несколько методов для более точного результата
""")
