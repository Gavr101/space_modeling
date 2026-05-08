# Space Modeling

Система моделирования движения малых спутников и определения орбит по шумным наблюдениям.

## Статус

Инициализирован каркас проекта с модульной архитектурой:

- `dynamics/` — высокоточная динамика и численное интегрирование;
- `estimation/` — генерация измерений, OD и оценка ковариации;
- `visualization/` — 3D визуализация орбит и области неопределенности;
- `experiments/` — валидационные сценарии и тесты;
- `data/` — TLE, эфемериды и наблюдения.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Ограничения

Проект намеренно не реализует вручную:

- системы времени;
- frame transforms;
- эфемериды и астрономические модели.

Для этого используются библиотеки TudatPy, Orekit, Astropy и SGP4.

## Demo notebook

Запуск демонстрации:

```bash
pip install -r requirements.txt
pip install -e .
jupyter notebook notebooks/orbit_propagation_demo.ipynb
```

Быстрый запуск в Google Colab: [Open in Colab](https://colab.research.google.com/github/Gavr101/space_modeling/blob/main/notebooks/orbit_propagation_demo.ipynb).

Для Google Colab в ноутбуке есть автоматическая установка зависимостей и клонирование репозитория.
