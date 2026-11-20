# dreambank-analysis

Анализ корпуса сновидений [DreamBank](https://dreambank.net/) (Adam Schneider, G. William Domhoff): лексика, эмоции, темы и статистическая устойчивость выводов.

## Что делает

Локальный HTML-снимок DreamBank разбирается в типизированные таблицы (`series`, `dreams`), по которым считаются восемь групп экспериментов:

| # | Раздел | Что проверяется |
|---|---|---|
| 1 | `IzzyEmotionsByAge` | Частота NRC-эмоций и VADER-тональность по возрастам дневника Izzy |
| 2 | `IzzyLengthVocabulary` | Длина записей и разнообразие словаря (MATTR-50) |
| 3 | `IzzyTopics` | TF-IDF + NMF, восемь тем и их вес по возрастам |
| 4 | `GenderComparison` | `college-f` vs `college-m`: Mann–Whitney, BH-поправка |
| 5 | `WordsByGender` | Частотный словарь по полу после стемминга |
| 6 | `Aggression` | Лексические маркеры агрессии (широкий/строгий словарь), Miami home vs lab, MDE, Fisher-power |
| 7 | `LengthAnger` | Пуассоновская модель частоты гнева с offset `log(token_count)`, HC3 и кластеры по блокам |
| 8 | `IzzyStability` | Устойчивость показателей внутри возрастных папок |
| 9 | `Cooccurrence` | Совместная встречаемость 8 категорий NRC с поправкой на длину (MH OR) |
| 10 | `BarbSanders` | Половины серии `b`, `b` vs `b2`, Spearman-контроль профиля |
| 11 | `NamesTopics` | Влияние имён собственных на темы Izzy |

Каждый эксперимент пишет csv/parquet в `results/DD_MM_YYYY_HH_MM/` и png в `figures/`. Все графики и метрики автоматически выгружаются в **Grafana** — дашборд с результатами прогонов доступен по адресу из конфигурации проекта.

## Установка

```bash
git clone <repo>
cd dreambank-analysis
uv sync

```
Требуется Python ≥ 3.12 и [uv](https://docs.astral.sh/uv/).

## Запуск

```bash
# 1. Скачать снимок HTML DreamBank (один раз)
python -c "from dreambank.scraping.scraper import scrape; scrape()"

# 2. Прогнать весь пайплайн
python -m dreambank

# 3. Тесты
pytest -q
```


## Структура
src/dreambank/
├── scraping/ # скачивание снимка DreamBank
├── parsing/ # HTML → TypedFrame (series, dreams)
├── analytics/ # эксперименты и статистика
│ ├── extended.py # чистые вычисления секции 11
│ └── extended_pipeline.py # обвязка для pipeline
├── common/ # TypedFrame, схемы, общие утилиты
└── pipeline.py # оркестрация: parse → experiments → save

notebooks/DreamPoly.ipynb # интерактивный вариант с графиками
tests/ # pytest



## Методы

- **Эмоции:** [VADER](https://github.com/cjhutto/vaderSentiment) (compound score) и [NRCLex](https://pypi.org/project/NRCLex/) (частоты на 1000 токенов).
- **Темы:** TF-IDF + [NMF](https://scikit-learn.org/stable/modules/decomposition.html#nmf), 8 компонентов.
- **Статистика:** Mann–Whitney, критерий Фишера, Мантель–Хэнзель по стратам длины, пуассоновская регрессия с offset, bootstrap по блокам, поправка Benjamini–Hochberg.
- **Мощность:** MDE для реальных объёмов серий, симуляция мощности критерия Фишера.

Словарные показатели — это **лексические маркеры**, а не события и не переживания. Сны одного дневника не независимы; где возможно, используются устойчивые ошибки, кластеры по блокам и bootstrap. При множественных проверках — поправка BH.

## Ограничения

Словарные оценки не разбирают сюжет. NMF реагирует на имена. Возраст, эпоха и стиль дневника меняются вместе. Результаты относятся к конкретным коллекциям DreamBank, а не ко всем людям.

## Публикации

Domhoff & Schneider (2008), [Studying dream content using the archive and search engine on DreamBank.net](https://doi.org/10.1016/j.concog.2008.06.010).
