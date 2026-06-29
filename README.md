# Fraud P2P Detection

Учебный ML-проект для обнаружения мошеннических P2P-переводов. Модель
классифицирует перевод как легитимный или мошеннический на основании данных о
самой операции и исторической активности отправителя и получателя.

В проекте используется `CatBoostClassifier`. При подготовке признаков
соблюдается временной порядок: текущая операция получает агрегаты только из
операций, произошедших до начала оцениваемого периода.

## Результаты

Метрики последнего запуска на отложенной test-выборке:

| Метрика для класса fraud | Значение |
|---|---:|
| Precision | 0.7665 |
| Recall | 0.7151 |
| F1-score | 0.7399 |
| ROC-AUC | 0.9999 |
| Average Precision | 0.7569 |

Матрица ошибок:

```text
TN = 246485    FP = 39
FN =     51    TP = 128
```

Порог классификации `0.6600` выбран на validation-выборке при ограничении
`recall >= 0.70`. Test-выборка при выборе порога не использовалась. Полный
отчёт сохраняется в [`reports/metrics.json`](reports/metrics.json).

## Данные

Для запуска необходимы два файла:

```text
data/raw/final_p2p_log.csv
data/raw/final_trans_log.csv
```

Основные поля P2P-журнала:

- `EventTime` — время перевода;
- `UserID` — отправитель;
- `RecipientID` — получатель;
- `Amount` — сумма;
- `Currency` — валюта;
- `IsFraud` — целевой признак.

Журнал обычных транзакций содержит информацию о пользователе, времени,
торговце, сумме, стране и успешности операции.

Исходные данные не хранятся в Git из-за размера. Их необходимо получить
отдельно и положить в `data/raw/`.

## Установка

Рекомендуется Python 3.10 или 3.11.

```bash
python -m venv .venv
```

Активация окружения в Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Запуск

### Обучение

```bash
python main.py train
```

Режим `train`:

1. Загружает и проверяет исходные данные.
2. Выполняет временное разбиение.
3. Строит признаки из предшествующей истории.
4. Обучает CatBoost.
5. Выбирает threshold на validation-выборке.
6. Сохраняет модель и threshold.
7. Оценивает модель на test-выборке.

Создаваемые артефакты:

```text
models/fraud_model_no_leakage.cbm
models/fraud_threshold.json
reports/metrics.json
```

### Оценка сохранённой модели

```bash
python main.py evaluate
```

или сокращённо:

```bash
python main.py
```

В этом режиме обучение не запускается. Код загружает модель и threshold,
формирует test-признаки и пересчитывает метрики. Перед запуском файлы модели и
порога должны находиться в каталоге `models/`.

## Apache Airflow

В проекте реализованы два DAG:

```text
fraud_model_training
validate_data → generate_features → train_model → select_threshold → evaluate_model

fraud_batch_inference
validate_inputs → generate_features → predict → save_predictions
```

Большие таблицы признаков не передаются через XCom. Задачи сохраняют их в
`data/processed/`, а через XCom передаются только пути к артефактам.

### Запуск Airflow через Docker

Необходимы Docker Desktop и Docker Compose. На Windows следует использовать
Linux containers/WSL2. Для Airflow рекомендуется выделить Docker не менее 4 ГБ
оперативной памяти, лучше 8 ГБ.

Сборка образа:

```bash
docker compose build
```

Запуск:

```bash
docker compose up
```

Airflow UI будет доступен по адресу:

```text
http://localhost:8080
```

Логин и пароль, созданные командой `airflow standalone`, можно найти в логах:

```bash
docker compose logs airflow
```

Контейнер использует Airflow 3.2.2. Каталоги `dags`, `data`, `models`, `reports`
и `logs` подключаются как volumes, поэтому результаты остаются в проекте.

### DAG обучения

Перед запуском положите исходные CSV в `data/raw/`, затем запустите
`fraud_model_training` в UI или командой:

```bash
docker compose exec airflow airflow dags trigger fraud_model_training
```

DAG проверяет входные файлы, строит train/validation/test-признаки, обучает
модель, выбирает threshold на validation и сохраняет test-метрики.

### DAG пакетного инференса

Сначала должен успешно завершиться DAG обучения. Новый P2P-батч без столбца
`IsFraud` положите сюда:

```text
data/inference/new_p2p.csv
```

Обязательные столбцы:

```text
EventTime, UserID, RecipientID, Amount, Currency
```

Запуск:

```bash
docker compose exec airflow airflow dags trigger fraud_batch_inference
```

Результат появится в:

```text
data/predictions/fraud_predictions.csv
```

К исходным столбцам добавляются:

- `fraud_score` — вероятность мошенничества;
- `is_fraud_pred` — решение по сохранённому threshold.

Пути можно изменить в `.env`; пример находится в `.env.example`.

## Временное разбиение и защита от утечки

P2P-журнал сортируется по `EventTime` и разделяется приблизительно в отношении
`80/10/10`:

```text
train → validation → test
```

Границы журнала обычных транзакций определяются по тем же timestamps, что и
границы P2P-журнала.

Для обучения первые 20% train используются как исторический warm-up. Остальная
часть train получает признаки только из warm-up:

```text
warm-up history → model train
train history   → validation
train + val     → test
```

Функция генерации признаков отдельно принимает текущие операции,
P2P-историю и историю обычных транзакций. Перед расчётом выполняется проверка,
что максимальное время истории строго меньше минимального времени текущего
периода. Это предотвращает использование будущих событий.

## Признаки

Модель использует 30 признаков, включая:

- сумму, валюту, отправителя и получателя текущего перевода;
- количество, сумму, среднее и стандартное отклонение прошлых переводов
  отправителя;
- аналогичные агрегаты получателя;
- число уникальных получателей и отправителей;
- число предыдущих переводов конкретной пары;
- статистику обычных транзакций пользователя;
- долю успешных операций;
- количество уникальных торговцев и стран;
- час, день недели, месяц и признак выходного дня;
- отношения текущей суммы к историческим средним и z-score.

Категориальные признаки:

```text
UserID, RecipientID, Currency
```

## Модель

Основные параметры CatBoost:

```text
iterations = 500
learning_rate = 0.05
depth = 6
loss_function = Logloss
eval_metric = AUC
random_seed = 42
```

Порог выбирается отдельно, потому что доля мошеннических операций очень мала и
обычная accuracy не отражает качество антифрода.

## Структура проекта

```text
fraud-p2p-detection/
├── dags/
│   ├── fraud_training_dag.py
│   └── fraud_inference_dag.py
├── data/
│   ├── raw/                 исходные данные
│   ├── interim/             промежуточные данные
│   ├── processed/           артефакты между Airflow-задачами
│   ├── inference/           новые P2P-батчи
│   └── predictions/         результаты инференса
├── models/                  локальные модели и threshold
├── reports/
│   ├── figures/             графики
│   └── metrics.json         метрики последнего запуска
├── src/
│   ├── config.py            пути и константы
│   ├── data.py              загрузка и временное разбиение
│   ├── features.py          генерация признаков
│   ├── pipeline.py          функции задач Airflow
│   ├── train.py             обучение CatBoost
│   ├── threshold.py         выбор порога
│   ├── evaluate.py          расчёт метрик
│   └── utils.py             сохранение модели и отчётов
├── main.py                  режимы train и evaluate
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Что не хранится в Git

В `.gitignore` исключены:

- исходные CSV;
- обученные модели и threshold;
- виртуальное окружение;
- Python-кэши;
- файлы IDE;
- служебный каталог `catboost_info`.

## GitLab CI

При каждом push и merge request GitLab запускает задачу `python_syntax`, которая
проверяет синтаксис `main.py`, модулей и DAG-файлов.

Полное обучение модели не запускается в CI: исходные данные и обученная модель
не хранятся в Git. Аналогичную проверку можно запустить локально:

```bash
python -m compileall -q main.py src dags
```
