# predict_price.py
import pandas as pd
import joblib
from datetime import datetime
import os

MODEL_PATH = "data/ship_price_model.pkl"
FEATURES_PATH = "data/ship_price_model_features.pkl"


def get_user_input():
    """Запрашиваем у пользователя данные нового судна"""
    print("\n=== Введите данные судна для предсказания цены ===")
    dwt = input("Дедвейт (dwt, например: 60000): ").replace(' ', '')
    year = int(input("Год постройки (year, например: 2018): "))
    ship_type = input("Тип судна (type, например: 1, 2, 3 — как в данных): ").strip()
    country = input("Страна постройки (country, например: 1, 2, 3 — как в данных): ").strip()
    date_str = input("Дата сделки (date, в формате ГГГГ-ММ-ДД, например: 2025-07-01): ").strip()

    return {
        'dwt': float(dwt),
        'year': year,
        'type': ship_type,
        'country': country,
        'date': date_str
    }


def main():
    print("📂 Загружаем модель...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена по пути: {MODEL_PATH}")
        print("Сначала запустите train_model.py для обучения модели!")
        return

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    # Получаем данные от пользователя
    user_data = get_user_input()

    # Подготавливаем данные для модели
    new_ship = pd.DataFrame({
        'dwt': [user_data['dwt']],
        'year': [user_data['year']],
        'date': [int(datetime.fromisoformat(user_data['date']).timestamp())]
    })

    # Добавляем столбцы для type и country (one-hot encoding)
    for col in feature_names:
        if col.startswith('type_'):
            new_ship[col] = [1 if col == f"type_{user_data['type']}" else 0]
        elif col.startswith('country_'):
            new_ship[col] = [1 if col == f"country_{user_data['country']}" else 0]

    # Добавляем недостающие столбцы (если вдруг)
    for col in feature_names:
        if col not in new_ship.columns:
            new_ship[col] = 0

    # Приводим к правильному порядку
    new_ship = new_ship[feature_names]

    # Предсказываем
    predicted_price = model.predict(new_ship)[0]

    print("\n" + "="*50)
    print("📈 ПРЕДСКАЗАНИЕ ЦЕНЫ")
    print("="*50)
    print(f"Тип судна:      {user_data['type']}")
    print(f"Дедвейт:        {user_data['dwt']:,}")
    print(f"Год постройки:  {user_data['year']}")
    print(f"Страна:         {user_data['country']}")
    print(f"Дата сделки:    {user_data['date']}")
    print("-"*50)
    print(f"💰 Предсказанная цена: ${predicted_price:,.2f}")
    print("="*50)


if __name__ == "__main__":
    main()