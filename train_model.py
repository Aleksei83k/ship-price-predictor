# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Путь к данным
DATA_PATH = "data/ships.csv"
MODEL_PATH = "data/ship_price_model.pkl"
FEATURES_PATH = "data/ship_price_model_features.pkl"


def detect_separator(file_path, sample_lines=5):
    """Определяет, какой разделитель используется в CSV: ',' или ';'"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ''.join([f.readline() for _ in range(sample_lines)])
        comma_count = sample.count(',')
        semicolon_count = sample.count(';')
        return ';' if semicolon_count > comma_count else ','
    except Exception as e:
        print(f"⚠️ Не удалось определить разделитель автоматически: {e}. Используем ',' по умолчанию.")
        return ','


def clean_numeric_column(series):
    """
    Очищает числовой столбец: удаляет пробелы, заменяет запятые на точки (если нужно),
    преобразует в float.
    """
    if series.dtype == 'object':
        # Удаляем пробелы внутри чисел: "10 500 000" → "10500000"
        series = series.str.replace(' ', '', regex=False)
        # Если есть запятые как десятичные разделители — заменяем на точки
        series = series.str.replace(',', '.', regex=False)
        # Преобразуем в числовой тип
        series = pd.to_numeric(series, errors='raise')
    return series


def main():
    print("🚀 Загружаем данные из файла:", DATA_PATH)

    # Проверяем, существует ли файл
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Файл {DATA_PATH} не найден. Проверьте путь и наличие файла.")

    # Определяем разделитель
    sep = detect_separator(DATA_PATH)
    print(f"🔍 Определён разделитель: '{sep}'")

    # Загружаем данные
    try:
        data = pd.read_csv(DATA_PATH, sep=sep, encoding='utf-8')
        print("✅ Файл успешно прочитан")
    except Exception as e:
        raise Exception(f"❌ Ошибка при чтении файла: {e}")

    # Очищаем названия столбцов: убираем пробелы, приводим к нижнему регистру
    data.columns = data.columns.str.strip().str.lower()
    print("📊 Столбцы после очистки:", data.columns.tolist())

    # Проверяем обязательные столбцы
    required_columns = ['type', 'dwt', 'year', 'country', 'date', 'price']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"❌ Отсутствуют обязательные столбцы: {missing_columns}. Проверьте файл {DATA_PATH}")

    # Проверяем типы данных до очистки
    print("\n🔍 Типы данных до преобразований:")
    print(data.dtypes)

    # Очищаем числовые столбцы: dwt и price
    print("\n🧮 Очищаем числовые столбцы 'dwt' и 'price' от пробелов...")
    try:
        data['dwt'] = clean_numeric_column(data['dwt'])
        data['price'] = clean_numeric_column(data['price'])
        print("✅ Столбцы 'dwt' и 'price' успешно преобразованы в числовой формат")
    except Exception as e:
        raise ValueError(f"❌ Ошибка при преобразовании числовых столбцов: {e}. Проверьте формат чисел в файле.")

    # Удаляем строки с пропущенными значениями
    print(f"\n🧹 Исходное количество строк: {len(data)}")
    data = data.dropna()
    print(f"🧹 После удаления NaN: {len(data)} строк")

    if len(data) == 0:
        raise ValueError("❌ Все строки содержали NaN — данные пусты!")

    # Преобразуем дату — с обработкой ошибок
    print("📅 Преобразуем столбец 'date'...")
    try:
        # Пробуем стандартное преобразование
        data['date'] = pd.to_datetime(data['date'], errors='raise').astype('int64') // 10**9
        print("✅ Дата успешно преобразована (стандартный формат)")
    except Exception as e:
        print(f"⚠️ Ошибка при стандартном преобразовании: {e}")
        print("💡 Пробуем альтернативные форматы...")

        # Пробуем другие форматы
        formats_to_try = [
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%Y.%m.%d'
        ]

        success = False
        for fmt in formats_to_try:
            try:
                data['date'] = pd.to_datetime(data['date'], format=fmt).astype('int64') // 10**9
                print(f"✅ Дата успешно преобразована с форматом: {fmt}")
                success = True
                break
            except:
                continue

        if not success:
            raise ValueError(f"❌ Не удалось преобразовать столбец 'date'. Проверьте формат дат в файле. Пример: 2025-07-01")

    # One-hot encoding для категориальных признаков
    print("🔤 Кодируем столбцы 'type' и 'country'...")
    try:
        # Убедимся, что type и country — строки, а не числа
        data['type'] = data['type'].astype(str)
        data['country'] = data['country'].astype(str)
        data = pd.get_dummies(data, columns=['type', 'country'], drop_first=True)
    except Exception as e:
        raise Exception(f"❌ Ошибка при кодировании категорий: {e}")

    # Проверяем, что целевой столбец 'price' существует
    if 'price' not in data.columns:
        raise ValueError("❌ Столбец 'price' исчез после кодирования. Проверьте исходные данные.")

    # Разделяем на признаки и целевую переменную
    X = data.drop('price', axis=1)
    y = data['price']

    print(f"\n📊 Размер данных после обработки: {X.shape[0]} строк, {X.shape[1]} признаков")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("❌ Данные пусты после обработки. Проверьте входной файл.")

    # Разделяем на обучающую и тестовую выборки
    print("🧩 Разделяем данные на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создаём и обучаем модель
    print("🤖 Обучаем модель Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise Exception(f"❌ Ошибка при обучении модели: {e}")

    # 🔄 Кросс-валидация на обучающей выборке
    print("\n🧪 Запускаем кросс-валидацию (5 фолдов)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # MAE
    cv_scores_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
    # RMSE
    cv_scores_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
    # R²
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

    print(f"📊 CV MAE: ${cv_scores_mae.mean():,.2f} ± ${cv_scores_mae.std():,.2f}")
    print(f"📊 CV RMSE: ${cv_scores_rmse.mean():,.2f} ± ${cv_scores_rmse.std():,.2f}")
    print(f"📊 CV R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")

    # 📊 Визуализация кросс-валидации
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # MAE
    ax[0].bar(range(1, len(cv_scores_mae)+1), cv_scores_mae, color='#3498db', edgecolor='black')
    ax[0].axhline(cv_scores_mae.mean(), color='red', linestyle='--', label=f'Среднее: ${cv_scores_mae.mean():,.0f}')
    ax[0].set_title('MAE по фолдам')
    ax[0].set_xlabel('Фолд')
    ax[0].set_ylabel('MAE ($)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # RMSE
    ax[1].bar(range(1, len(cv_scores_rmse)+1), cv_scores_rmse, color='#e74c3c', edgecolor='black')
    ax[1].axhline(cv_scores_rmse.mean(), color='red', linestyle='--', label=f'Среднее: ${cv_scores_rmse.mean():,.0f}')
    ax[1].set_title('RMSE по фолдам')
    ax[1].set_xlabel('Фолд')
    ax[1].set_ylabel('RMSE ($)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # R²
    ax[2].bar(range(1, len(cv_scores_r2)+1), cv_scores_r2, color='#2ecc71', edgecolor='black')
    ax[2].axhline(cv_scores_r2.mean(), color='red', linestyle='--', label=f'Среднее: {cv_scores_r2.mean():.3f}')
    ax[2].set_title('R² по фолдам')
    ax[2].set_xlabel('Фолд')
    ax[2].set_ylabel('R²')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/cv_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 График кросс-валидации сохранён: data/cv_results.png")
    plt.show()

    # Оцениваем точность на отложенной выборке
    print("\n📈 Оцениваем качество модели на тестовой выборке...")
    y_pred = model.predict(X_test)

    # MAE — Средняя абсолютная ошибка
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE — Корень из средней квадратичной ошибки
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE — Средняя абсолютная процентная ошибка (в процентах)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

    # R² — Коэффициент детерминации
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Результаты на тестовой выборке:")
    print(f"Средняя абсолютная ошибка (MAE): ${mae:,.2f}")
    print(f"Корень из средней квадратичной ошибки (RMSE): ${rmse:,.2f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"Коэффициент детерминации (R²): {r2:.3f}")

    if r2 < 0:
        print("⚠️ Внимание: R² отрицательный — модель работает хуже, чем просто среднее значение. Проверьте данные!")

    # 🎨 Построение диагностики модели
    print("\n🖼️ Строим графики истинных vs предсказанных цен...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: Истинные vs Предсказанные цены
    ax1.scatter(y_test, y_pred, alpha=0.7, color='#3498db', edgecolors='w', s=60)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальное предсказание')
    ax1.set_xlabel('Истинная цена ($)', fontsize=12)
    ax1.set_ylabel('Предсказанная цена ($)', fontsize=12)
    ax1.set_title('Истинные vs Предсказанные цены', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Распределение ошибок
    errors = y_test - y_pred
    ax2.hist(errors, bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Ошибка (Истинная - Предсказанная)', fontsize=12)
    ax2.set_ylabel('Количество', fontsize=12)
    ax2.set_title('Распределение ошибок', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/model_diagnostics.png', dpi=150, bbox_inches='tight')
    print("📊 Диагностические графики сохранены: data/model_diagnostics.png")
    plt.show()

    # Сохраняем модель и список признаков
    print(f"\n💾 Сохраняем модель в {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
        print("🎉 Модель и список признаков успешно сохранены!")
    except Exception as e:
        raise Exception(f"❌ Ошибка при сохранении модели: {e}")

    # Дополнительно: выводим 5 самых важных признаков
    print("\n🔝 Топ-5 важных признаков:")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(feature_importances.head().items(), 1):
        print(f"{i}. {feat}: {imp:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("💡 Совет: проверьте формат данных, наличие столбцов, разделитель в CSV и путь к файлу.")
        exit(1)
