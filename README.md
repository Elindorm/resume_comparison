
### 1. Загрузка и предобработка данных

В начале ноутбука происходит загрузка данных резюме и предобработка. Данные очищаются и подготавливаются для дальнейшего анализа и моделирования:

```python
import pandas as pd

# Загрузка данных
df_resumes = pd.read_csv('resumes.csv')

# Предобработка данных
df_resumes['experience'] = df_resumes['experience'].apply(lambda x: int(x.split()[0]) if pd.notna(x) else 0)
df_resumes.dropna(subset=['name', 'birth_date'], inplace=True)
df_resumes['birth_date'] = pd.to_datetime(df_resumes['birth_date'])
df_resumes['birth_date'] = (pd.Timestamp.now() - df_resumes['birth_date']).astype('<m8[Y]')

# Кодирование категориальных переменных
df_resumes = pd.get_dummies(df_resumes, columns=['city', 'gender'])
```

### 2. Разделение данных на обучающую и тестовую выборки

Данные разделяются на обучающую и тестовую выборки, и проводится стандартизация признаков:

```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_resumes, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols_to_scale = ['birth_date', 'experience']
df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])
```

### 3. Обработка несбалансированных данных с использованием SMOTE

Для балансировки данных используется метод SMOTE:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X = df_train.drop(columns=['status'])
y = df_train['status']

X_resampled, y_resampled = smote.fit_resample(X, y)
df_train_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_train_resampled['status'] = y_resampled
```

### 4. Обучение моделей и оценка их эффективности

Обучаются различные модели классификации, такие как Random Forest и XGBoost, и оценивается их точность:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
rf_model.fit(X_train, y_train)

accuracy = rf_model.score(X_test, y_test)
print("Accuracy:", accuracy)

from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=10, learning_rate=0.05, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### 5. Обработка тестовых данных и предсказание результатов

Тестовые данные обрабатываются, и для них предсказываются результаты. Выводятся UUID кандидатов, которые были приняты на работу:

```python
X_test_processed = df_test.drop('uuid', axis=1)

predictions = rf_model.predict(X_test_processed)

accepted_indices = [idx for idx, pred in enumerate(predictions) if pred == 1]

for idx in accepted_indices:
    print(df_test.loc[idx, 'uuid'])

with open('accepted_candidates.txt', 'w') as f:
    for idx in accepted_indices:
        f.write(f"{df_test.loc[idx, 'uuid']}\n")

print("Общее количество принятых кандидатов:", len(accepted_indices))
```

### Заключение

Этот ноутбук включает в себя весь процесс обработки данных резюме, начиная с загрузки и предобработки данных, и заканчивая обучением моделей машинного обучения и предсказанием результатов для тестового набора данных.
