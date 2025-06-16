# mts_mlops_2

Датасеты предоставлены в рамках соревнования https://www.kaggle.com/competitions/teta-ml-1-2025

Система для обнаружения мошеннических транзакций в реальном времени с использованием ML-модели и Kafka для потоковой обработки данных.

## 🏗️ Архитектура

Компоненты системы:
1. **`interface`** (Streamlit UI):
   
   Создан для удобной симуляции потоковых данных с транзакциями. Реальный продукт использовал бы прямой поток данных из других систем.
    - Имитирует отправку транзакций в Kafka через CSV-файлы.
    - Генерирует уникальные ID для транзакций.
    - Загружает транзакции отдельными сообщениями формата JSON в топик kafka `transactions`.
    
2. **`fraud_detector`** (ML Service):
   - Загружает предобученную модель CatBoost (`my_model.cbm`).
   - Выполняет препроцессинг данных:
     - Извлечение временных признаков
   - Производит скоринг с порогом 0.5.
   - Выгружает результат скоринга в топик kafka `scoring`

3. **Kafka Infrastructure**:
   - Zookeeper + Kafka брокер
   - `kafka-setup`: автоматически создает топики `transactions` и `scoring`
   - Kafka UI: веб-интерфейс для мониторинга сообщений (порт 8080)
  
4. **PostgreSQL** (Database):
   - Postgres 16, база fraud, том pgdata.
   - `postgres/init.sql` создаёт таблицу fraud_scores (UUID, score, fraud_flag, timestamp).

5. **score-sink** (Sink Service):
   - На aiokafka и asyncpg, консьюмирует топик scoring.
   - Нормализует поля и записывает в Postgres fraud_scores.

6. **interface** (Streamlit):
   - Кнопка «Посмотреть результаты» выводит:
      - 10 последних фродовых транзакций (fraud_flag == 1).
      - Гистограмма распределения score для последних 100 записей.

## 🚀 Быстрый старт

### Требования
- Docker 20.10+
- Docker Compose 2.0+

### Запуск
```bash
git clone https://github.com/ArtyomRukavitsa/mts_mlops_2.git
cd mts_mlops_2

# Сборка и запуск всех сервисов
docker-compose up --build
```
После запуска:
- **Streamlit UI**: http://localhost:8501
- **Kafka UI**: http://localhost:8080
- **Логи сервисов**: 
  ```bash
  docker-compose logs <service_name>  # Например: fraud_detector, kafka, interface

## 🛠️ Использование

### 1. Загрузка данных:

 - Загрузите CSV через интерфейс Streamlit. Для тестирования работы проекта используется файл формата `test.csv` из соревнования https://www.kaggle.com/competitions/teta-ml-1-2025
 - Пример структуры данных:
    ```csv
    transaction_time,amount,lat,lon,merchant_lat,merchant_lon,gender,...
    2023-01-01 12:30:00,150.50,40.7128,-74.0060,40.7580,-73.9855,M,...
    ```
 - Для первых тестов рекомендуется загружать небольшой семпл данных (до 100 транзакций) за раз, чтобы исполнение кода не заняло много времени.
 - Далее в UI можно нажать кнопку Посмотреть результаты, которая выведет таблицу и гистограмму.
 - Важно отметить, что при тестировании программы на небольших тестировочных данных фродовых транзакций моя модель не обнаружила, поэтому для проверки корректности работы таблицы в UI можно создать искусственную фродовую транзакцию:
   ```bash
   docker compose exec db psql -U fraud -d fraud -c "INSERT INTO fraud_scores (transaction_id, score, fraud_flag) VALUES ('550e8400-e29b-41d4-a716-446655440000', 0.95, 1);"

### 2. Мониторинг:
 - **Kafka UI**: Просматривайте сообщения в топиках transactions и scoring
 - **Логи обработки**: /app/logs/service.log внутри контейнера fraud_detector

### 3. Результаты:
 - Скоринговые оценки пишутся в топик scoring в формате:
    ```json
    {
    "score": 0.995, 
    "fraud_flag": 1, 
    "transaction_id": "d6b0f7a0-8e1a-4a3c-9b2d-5c8f9d1e2f3a"
    }
    ```
 - При нажатии кнопки Посмотреть результаты отображаются таблица и гистограмма.
## Структура проекта
```
.
├── fraud_detector/
│   ├── preprocessing.py    # Логика препроцессинга
│   ├── scorer.py           # ML-модель и предсказания
│   ├── app.py              # Kafka Consumer/Producer
│   └── Dockerfile
├── interface/
│   └── app.py              # Streamlit UI
├─ score_sink/              # Сервис-синк (читает скоринг, пишет в Postgres)
│   └── consumer.py 
└─ postgres/
│  └── init.sql             # Инициализация таблицы fraud_scores
├── docker-compose.yaml     # Для поднятия всех сервисов
├── new_small.csv           # Пример тестировочных данных
└── README.md
```

## Настройки Kafka
```yml
Топики:
- transactions (входные данные)
- scoring (результаты скоринга)

Репликация: 1 (для разработки)
Партиции: 3
```
