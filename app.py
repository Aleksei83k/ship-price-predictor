# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Пути к модели и признакам
MODEL_PATH = "data/ship_price_model.pkl"
FEATURES_PATH = "data/ship_price_model_features.pkl"

# Загружаем модель и список признаков
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}. Сначала обучите модель через train_model.py")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из формы
        dwt = float(request.form['dwt'])
        year = int(request.form['year'])
        ship_type = request.form['type']
        country = request.form['country']
        date_str = request.form['date']

        # Преобразуем дату
        date_ts = int(datetime.fromisoformat(date_str).timestamp())

        # Создаём DataFrame
        new_ship = pd.DataFrame({
            'dwt': [dwt],
            'year': [year],
            'date': [date_ts]
        })

        # One-hot encoding
        for col in feature_names:
            if col.startswith('type_'):
                new_ship[col] = [1 if col == f"type_{ship_type}" else 0]
            elif col.startswith('country_'):
                new_ship[col] = [1 if col == f"country_{country}" else 0]

        # Добавляем недостающие столбцы
        for col in feature_names:
            if col not in new_ship.columns:
                new_ship[col] = 0

        # Приводим к правильному порядку
        new_ship = new_ship[feature_names]

        # Предсказываем
        predicted_price = model.predict(new_ship)[0]

        return render_template('index.html',
                             prediction=f"${predicted_price:,.2f}",
                             dwt=dwt,
                             year=year,
                             ship_type=ship_type,
                             country=country,
                             date=date_str)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
