# InternVL3.5 Demo

Простое web-приложение для демонстрации возможностей мультимодальной модели [InternVL3.5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B) (Visual Question Answering, Captioning, Object Detection).

## Возможности (Use Cases)

1.  **VQA & Captioning / Image Description**
    *   Загрузите изображение и задайте вопрос по его содержимому.
    *   Или попросите модель описать изображение.
    *   Поддерживается диалог (история сообщений сохраняется в рамках сессии).
2.  **Поиск объектов (Visual Grounding / Object Detection)**
    *   Загрузите изображение и введите текстовое описание объекта (например, "cat", "red car").
    *   Модель вернет изображение с выделенными bounding box'ами и JSON-отчет с координатами.
    *   Результаты (изображение и JSON) доступны для скачивания.

Приложение валидирует входные данные и отображает понятные ошибки.

## Запуск через Docker (Рекомендуется)

Приложение запускается в контейнере Docker. Поддерживаются режимы CPU и GPU.

### Предварительные требования
*   Docker Engine
*   Docker Compose v2

### 1. Сборка и запуск (CPU)

Это универсальный вариант, работает на большинстве машин (Linux, macOS, Windows).

```bash
docker compose up --build internvl-demo-cpu
```

### 2. Сборка и запуск (GPU)

Требуется NVIDIA GPU и драйверы с поддержкой CUDA.

```bash
docker compose up --build internvl-demo-gpu
```

При первом запуске приложение автоматически скачает веса модели (~2.5GB) с Hugging Face Hub в директорию `./data/model_cache`. Последующие запуски будут использовать этот локальный кэш и работать **офлайн**.

### 3. Доступ к приложению

После успешного запуска откройте в браузере:
**http://localhost:7860**

## Конфигурация

Все настройки можно менять через переменные окружения в файле `.env` или передавать их при запуске.

### Основные параметры

| Переменная | Значение по умолчанию | Описание |
| :--- | :--- | :--- |
| `PORT` | `7860` | Порт, на котором будет доступно web-приложение. |
| `DEVICE` | `auto` | Устройство для инференса: `cpu`, `cuda`, или `auto` (автоопределение). |
| `MODEL_NAME` | `OpenGVLab/InternVL3_5-1B` | Имя модели на HuggingFace Hub или путь к локальной директории внутри контейнера. |
| `HF_CACHE_DIR` | `/cache` | Путь внутри контейнера, куда монтируется кэш моделей. |

### Как изменить порт

В файле `.env`:
```env
PORT=8080
```
Или при запуске:
```bash
PORT=8080 docker compose up internvl-demo-cpu
```
Приложение будет доступно по адресу `http://localhost:8080`.

### Как изменить режим (CPU/GPU)

Для запуска на CPU используйте сервис `internvl-demo-cpu`.
Для запуска на GPU используйте сервис `internvl-demo-gpu` (требует `nvidia-container-toolkit`).

### Как изменить размер модели

Для использования другой модели (например, SmolVLM2 или более крупной InternVL), измените переменную `MODEL_NAME` в `.env`:

```env
MODEL_NAME=OpenGVLab/InternVL2-2B
```
*Примечание: может потребоваться адаптация кода, если новая модель использует другой API процессора.*

### Монтирование весов модели с хоста

В `docker-compose.yml` уже настроено монтирование директории для кэширования весов:

```yaml
volumes:
  - ./data/model_cache:/cache
```

*   При первом старте веса скачиваются в `./data/model_cache` на хосте.
*   Чтобы использовать заранее скачанные веса, просто поместите их в эту директорию, сохраняя структуру huggingface (или измените `MODEL_NAME` на путь к конкретной папке, если скачивали вручную, например `snapshot_download`).

## Техническое описание модели

Используемая модель: **InternVL3.5-1B**

*   **Architecture**: Мультимодальная LLM, состоящая из Vision Encoder (InternViT) и LLM (Qwen2-0.5B), соединенных через MLP проектор.
*   **Capabilities**: OCR, VQA, Visual Grounding.
*   **Paper/Report**: [InternVL: Scaling up Vision Foundation Models and Multi-Modal LLMs](https://arxiv.org/abs/2312.14238) / [Hugging Face Model Card](https://huggingface.co/OpenGVLab/InternVL3_5-1B)

## Структура проекта

*   `app/` - Исходный код приложения (FastAPI/Gradio).
*   `Dockerfile` - Инструкции по сборке образа.
*   `docker-compose.yml` - Конфигурация сервисов (CPU/GPU).
*   `pyproject.toml` - Зависимости Python.
*   `data/model_cache/` - Локальное хранилище весов (создается автоматически).
