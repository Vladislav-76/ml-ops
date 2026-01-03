from typing import List

import pandas as pd
import treelite
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


# Загружаем данные
url: str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df: pd.DataFrame = pd.read_csv(url)

# Обрабатываем пропущенные значения
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Создаем новые признаки
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Кодируем категориальные переменные
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Выбираем признаки для модели
features: List[str] = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X: pd.DataFrame = df[features]
y: pd.Series = df['Survived']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Экспортируем модель через treelite
tl_model = treelite.sklearn.import_model(model)
tl_model.serialize("checkpoint.tl")
