# Product Category Classification

Предсказание категории товара по текстовым полям (name, description, model, type_prefix, vendor), URL и изображению.

## Структура проекта

```
MLOPSP/
├── src/data/           Python-библиотека обработки данных
│   ├── loader.py       Загрузка parquet/csv файлов
│   ├── preprocessor.py Очистка текста, конкатенация полей
│   └── features.py     TF-IDF извлечение признаков
├── scripts/
│   ├── eda.py          EDA — статистики и графики → reports/
│   └── train.py        Обучение модели с MLflow-трекингом
├── configs/
│   └── config.yaml     Все гиперпараметры эксперимента
├── reports/            Графики EDA (генерируются автоматически)
├── models/             Обученные модели (генерируются автоматически)
├── requirements.txt    Зафиксированные версии зависимостей
└── setup.py            Установка src как пакета
```

## Быстрый старт

### 1. Создание окружения

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Данные

Положить в корень проекта:
```
train.parquet.snappy
test.parquet.snappy
tree.csv
```

### 4. EDA

```bash
python scripts/eda.py
```

Создаёт в `reports/`:
- `category_distribution.png` — гистограмма и топ-20 категорий
- `text_lengths.png` — длины текстов по полям
- `top_level_categories.png` — объём по топ-уровням иерархии

### 5. Обучение

```bash
python scripts/train.py
```

Запускает полный пайплайн:
1. Загрузка данных из путей в `configs/config.yaml`
2. Очистка и конкатенация текстовых полей
3. TF-IDF векторизация
4. Обучение LogisticRegression
5. Логирование параметров и метрик в MLflow
6. Сохранение модели в `models/`

Вывод в консоль:
```
val_accuracy:    0.9535
val_f1_macro:    0.3281
val_f1_weighted: 0.9404
Model saved to: models/model.pkl
```

### 6. MLflow UI

```bash
mlflow ui
```

Открыть в браузере: `http://localhost:5000`

Там доступны все запуски с параметрами, метриками и артефактами.

## Конфигурация

Все параметры в `configs/config.yaml`:

```yaml
features:
  text_columns: ["name", "description", "model", "type_prefix", "vendor"]
  max_features: 100000   # размер словаря TF-IDF
  ngram_range: [1, 2]    # униграммы + биграммы
  min_df: 2              # минимальная частота токена

model:
  C: 10.0                # регуляризация LogReg (меньше → сильнее)
  max_iter: 1000
  solver: "saga"

training:
  test_size: 0.2
  random_state: 42
  mlflow_experiment: "product_categorization"
```

Изменить параметр → перезапустить `scripts/train.py` → новый run в MLflow.

## Использование src как библиотеки

```python
from src.data import load_train, combine_text_features, TextFeatureExtractor

train = load_train("train.parquet.snappy")
texts = combine_text_features(train, ["name", "description", "vendor"])

extractor = TextFeatureExtractor(max_features=50000, ngram_range=(1, 2))
X = extractor.fit_transform(texts)
```

## Результаты baseline

| Метрика | Значение |
|---|---|
| val_accuracy | 0.9535 |
| val_f1_weighted | 0.9404 |
| val_f1_macro | 0.3281 |

Модель: TF-IDF (100k фич, bigrams) + LogisticRegression (C=10, saga).

Низкий F1-macro объясняется сильным дисбалансом: 65% классов имеют < 50 примеров в train.
