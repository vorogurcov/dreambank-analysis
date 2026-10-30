"""Run the system on the local HTML snapshot and write a timestamped result folder."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from dreambank.parsing import ParsedArchive, parse_html, read_archive, write_tables
from dreambank.scraping.scraper import default_html_root

logger = logging.getLogger()
SPB = ZoneInfo("Europe/Moscow")
_EXPERIMENTS = (
    ("IzzyEmotionsByAge", "эмоции по возрасту"),
    ("IzzyLengthVocabulary", "длина и словарь"),
    ("IzzyTopics", "темы"),
    ("GenderComparison", "сравнение по полу"),
    ("WordsByGender", "слова по полу"),
)


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
    _run_experiments(processed, output, figures)
    logger.info("Записал результаты: %s", output)
    return archive


def _run_experiments(processed: Path, output: Path, figures: Path) -> None:
    workers = min(len(_EXPERIMENTS), os.cpu_count() or 1)
    if workers == 1:
        archive = read_archive(processed)
        for name, detail in _EXPERIMENTS:
            logger.info("Запустил эксперимент %s: %s", name, detail)
            _execute(name, archive, output, figures)
            logger.info("Закончил эксперимент %s", name)
        return
    logger.info("Считаю эксперименты параллельно, процессов: %s", workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {}
        for name, detail in _EXPERIMENTS:
            logger.info("Запустил эксперимент %s: %s", name, detail)
            running[pool.submit(_run_experiment, name, processed, output, figures)] = name
        for finished in as_completed(running):
            finished.result()
            logger.info("Закончил эксперимент %s", running[finished])


def _run_experiment(name: str, processed: Path, output: Path, figures: Path) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    _execute(name, read_archive(processed), Path(output), Path(figures))
    return name


def _execute(name: str, archive: ParsedArchive, output: Path, figures: Path) -> None:
    if name == "IzzyEmotionsByAge":
        emotions = IzzyEmotionsByAge(IzzyAgeConfig(figure_path=figures / "01_emotions_by_age.png"))
        emotions.prepare(archive)
        emotions.analyze()
        emotions.to_csv(output / "izzy_by_age.csv")
        _close(emotions.plot())
        return
    if name == "IzzyLengthVocabulary":
        length = IzzyLengthVocabulary(IzzyAgeConfig(figure_path=figures / "02_length_vocabulary.png"))
        length.prepare(archive)
        length.analyze()
        _close(length.plot())
        return
    if name == "IzzyTopics":
        topics = IzzyTopics(IzzyTopicsConfig(figure_path=figures / "03_topics_by_age.png"))
        topics.prepare(archive)
        topics.analyze()
        topics.to_csv(output / "topics_by_age.csv")
        topics.get_results().terms.to_csv(output / "topic_terms.csv", index=False, encoding="utf-8-sig")
        _close(topics.plot())
        return
    if name == "GenderComparison":
        gender = GenderComparison(GenderConfig(figure_path=figures / "04_gender_metrics.png"))
        gender.prepare(archive)
        gender.analyze()
        gender.to_csv(output / "gender_comparison.csv")
        _close(gender.plot())
        return
    if name == "WordsByGender":
        words = WordsByGender(WordsByGenderConfig(figure_path=figures / "05_words_by_gender.png"))
        words.prepare(archive)
        words.analyze()
        words.to_csv(output / "words_by_gender.csv")
        _close(words.plot())
        return
    raise ValueError(f"Unknown experiment: {name}")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()


def _close(figure: object) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
