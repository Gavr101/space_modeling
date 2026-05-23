# Space Modeling

Система моделирования движения малых спутников и определения орбит по шумным наблюдениям.

## Статус

Инициализирован каркас проекта с модульной архитектурой:

- `dynamics/` — высокоточная динамика и численное интегрирование;
- `estimation/` — генерация измерений, OD и оценка ковариации;
- `visualization/` — 3D визуализация орбит и области неопределенности;
- `experiments/` — валидационные сценарии и тесты;
- `data/` — TLE, эфемериды и наблюдения.

## Быстрый старт (локально)

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

## Demo notebooks

- Orbit propagation demo: `notebooks/orbit_propagation_demo.ipynb`
- Prediction error growth demo: `notebooks/prediction_error_growth_demo.ipynb`

Локальный запуск:

```bash
pip install -r requirements.txt
pip install -e .
jupyter notebook notebooks/orbit_propagation_demo.ipynb
```

## Google Colab

Оба ноутбука адаптированы для запуска в Colab:

- [Open in Colab: orbit_propagation_demo](https://colab.research.google.com/github/Gavr101/space_modeling/blob/main/notebooks/orbit_propagation_demo.ipynb)
- [Open in Colab: prediction_error_growth_demo](https://colab.research.google.com/github/Gavr101/space_modeling/blob/main/notebooks/prediction_error_growth_demo.ipynb)

Что делает установочная ячейка в Colab:

1. Клонирует репозиторий в `/content/space_modeling` (если ещё не клонирован).
2. Обновляет `pip`.
3. Устанавливает зависимости из `requirements.txt`.
4. Устанавливает пакет проекта (`pip install -e .`).
5. Пытается установить `nrlmsise00` (опционально): если установка не удалась, ноутбук продолжит работу с fallback-моделью плотности атмосферы.

Это важно, потому что в `dynamics.propagator` модель сопротивления атмосферы использует NRLMSISE-00 при наличии пакета, а при его отсутствии автоматически переключается на резервную экспоненциальную модель.

Примечание по стабильности Colab: в демо-ноутбуках принудительно отключён флаг `nrlmsise00_atmosphere` для расчёта траекторий, чтобы избежать падений из-за несовместимых версий `nrlmsise00` в разных окружениях. При этом аэродинамическое сопротивление остаётся включённым и использует fallback-модель плотности.
