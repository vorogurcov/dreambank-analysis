"""Run the system on the local HTML snapshot and write a timestamped result folder."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

from dreambank.analytics import (
    GenderComparison,
    GenderConfig,
    IzzyAgeConfig,
    IzzyEmotionsByAge,
    IzzyLengthVocabulary,
    IzzyTopics,
    IzzyTopicsConfig,
    WordsByGender,
    WordsByGenderConfig,
)
from dreambank.parsing import ParsedArchive, parse_html, write_tables
from dreambank.scraping.scraper import default_html_root

logger = logging.getLogger()
SPB = ZoneInfo("Europe/Moscow")


def results_root() -> Path:
    return Path(__file__).resolve().parent / "results"


def default_output_dir() -> Path:
    """``src/dreambank/results/DD_MM_YYYY_HH_MM`` in Saint Petersburg time."""
    stamp = datetime.now(SPB).strftime("%d_%m_%Y_%H_%M")
    return results_root() / stamp


@dataclass(frozen=True)
class PipelineConfig:
    """Local snapshot in, a new timestamped folder under ``src/dreambank/results`` out."""

    html_root: Path = field(default_factory=default_html_root)
    output: Path = field(default_factory=default_output_dir)


def run(config: PipelineConfig | None = None) -> ParsedArchive:
    """Parse the snapshot and write the same tables and figures as the notebook."""
    settings = config if config is not None else PipelineConfig()
    output = Path(settings.output)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    archive = parse_html(settings.html_root)
    logger.info(
        "Собрал информацию из %s: %s серий, %s записей, %s уникальных текстов",
        settings.html_root,
        len(archive.series),
        len(archive.dreams),
        len(archive.unique_dreams),
    )
    processed = output / "processed"
    write_tables(archive, processed)
    logger.info("Обработал архив: %s", processed)

    logger.info("Запустил эксперимент IzzyEmotionsByAge: эмоции по возрасту")
    emotions = IzzyEmotionsByAge(IzzyAgeConfig(figure_path=figures / "01_emotions_by_age.png"))
    emotions.prepare(archive)
    emotions.analyze()
    emotions.to_csv(output / "izzy_by_age.csv")
    _close(emotions.plot())
    logger.info("Закончил эксперимент IzzyEmotionsByAge")

    logger.info("Запустил эксперимент IzzyLengthVocabulary: длина и словарь")
    length = IzzyLengthVocabulary(IzzyAgeConfig(figure_path=figures / "02_length_vocabulary.png"))
    length.prepare(archive)
    length.analyze()
    _close(length.plot())
    logger.info("Закончил эксперимент IzzyLengthVocabulary")

    logger.info("Запустил эксперимент IzzyTopics: темы")
    topics = IzzyTopics(IzzyTopicsConfig(figure_path=figures / "03_topics_by_age.png"))
    topics.prepare(archive)
    topics.analyze()
    topics.to_csv(output / "topics_by_age.csv")
    topics.get_results().terms.to_csv(output / "topic_terms.csv", index=False, encoding="utf-8-sig")
    _close(topics.plot())
    logger.info("Закончил эксперимент IzzyTopics")

    logger.info("Запустил эксперимент GenderComparison: сравнение по полу")
    gender = GenderComparison(GenderConfig(figure_path=figures / "04_gender_metrics.png"))
    gender.prepare(archive)
    gender.analyze()
    gender.to_csv(output / "gender_comparison.csv")
    _close(gender.plot())
    logger.info("Закончил эксперимент GenderComparison")

    logger.info("Запустил эксперимент WordsByGender: слова по полу")
    words = WordsByGender(WordsByGenderConfig(figure_path=figures / "05_words_by_gender.png"))
    words.prepare(archive)
    words.analyze()
    words.to_csv(output / "words_by_gender.csv")
    _close(words.plot())
    logger.info("Закончил эксперимент WordsByGender")
    logger.info("Записал результаты: %s", output)
    return archive


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()


def _close(figure: object) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
