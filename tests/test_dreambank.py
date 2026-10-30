import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from dreambank import (
    DreamsFrame,
    GenderComparison,
    GenderConfig,
    IzzyAgeConfig,
    IzzyEmotionsByAge,
    IzzyLengthVocabulary,
    IzzyTopics,
    IzzyTopicsConfig,
    ScrapeConfig,
    SeriesFrame,
    WordsByGender,
    WordsByGenderConfig,
    parse_html,
    read_archive,
    write_tables,
)
from dreambank.analytics.constants import EMOTIONS, TOPIC_LABELS
from dreambank.analytics.frames import FEATURE_COLUMNS, DreamFeaturesFrame, TopicsByAgeFrame
from dreambank.common.frame import FrameSchemaError, column_schema
from dreambank.parsing.html import _reject_bad_dreams
from dreambank.scraping.scraper import is_safe_series_id, series_ids_from_html

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "notebooks" / "data" / "raw" / "html"
RESULTS = ROOT / "notebooks" / "results" / "DreamPoly"
PROCESSED = RESULTS / "processed"


def assert_csv(frame: pd.DataFrame, path: Path, *, rtol: float = 1e-8, atol: float = 1e-8, rename: dict[str, str] | None = None) -> None:
    expected = pd.read_csv(path, encoding="utf-8-sig")
    if rename:
        expected = expected.rename(columns=rename)
    got = pd.DataFrame(frame).reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, rtol=rtol, atol=atol)


class SchemaTests(unittest.TestCase):
    def test_column_order(self) -> None:
        self.assertEqual(
            [spec.name for spec in column_schema(SeriesFrame)],
            [
                "series_id",
                "language",
                "series_title",
                "reported_count",
                "series_years_raw",
                "dreamer_sex_raw",
                "parsed_count",
            ],
        )
        feature_names = [spec.name for spec in column_schema(DreamFeaturesFrame)]
        self.assertEqual(tuple(feature_names[-len(FEATURE_COLUMNS) :]), FEATURE_COLUMNS)
        topic_names = [spec.name for spec in column_schema(TopicsByAgeFrame)]
        self.assertEqual(topic_names, ["age_group", *TOPIC_LABELS])

    def test_empty_frames_have_declared_dtypes(self) -> None:
        series = SeriesFrame.blank()
        self.assertTrue(series.empty)
        self.assertEqual(len(series), 0)
        self.assertTrue(pd.api.types.is_integer_dtype(series["reported_count"]))
        self.assertEqual(len(DreamsFrame.blank()), 0)

    def test_rejects_extra_missing_dtype_and_literal(self) -> None:
        good = pd.DataFrame(
            {
                "series_id": ["alta"],
                "language": ["en"],
                "series_title": ["Alta"],
                "reported_count": [1],
                "series_years_raw": ["1985"],
                "dreamer_sex_raw": ["female"],
                "parsed_count": [1],
            }
        )
        built = SeriesFrame.build(good)
        self.assertFalse(built.empty)
        self.assertIn("alta", built.to_string())
        extra = good.copy()
        extra["extra"] = ["x"]
        with self.assertRaises(FrameSchemaError):
            SeriesFrame.build(extra)
        missing = good.drop(columns=["parsed_count"])
        with self.assertRaises(FrameSchemaError):
            SeriesFrame.build(missing)
        wrong = good.copy()
        wrong["reported_count"] = ["1"]
        with self.assertRaises(FrameSchemaError):
            SeriesFrame.build(wrong)
        language = good.copy()
        language["language"] = ["fr"]
        with self.assertRaises(FrameSchemaError):
            SeriesFrame.build(language)

    def test_dream_rules(self) -> None:
        dreams = DreamsFrame.build(
            pd.DataFrame(
                {
                    "dream_id": ["alta:1", "alta:1"],
                    "series_id": ["alta", "alta"],
                    "dream_number_raw": ["1", "1"],
                    "header_metadata_raw": ["", ""],
                    "text": ["hello", "hello"],
                    "word_count_reported": [1, 1],
                    "language": ["en", "en"],
                    "dreamer_sex_raw": ["female", "female"],
                    "series_years_raw": ["1985", "1985"],
                    "source_file": ["data/raw/html/alta/dreams.html", "data/raw/html/alta/dreams.html"],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "Repeated dream_id"):
            _reject_bad_dreams(dreams)
        dreams.loc[0, "text"] = ""
        dreams.loc[1, "dream_id"] = "alta:2"
        with self.assertRaisesRegex(ValueError, "Empty dream text"):
            _reject_bad_dreams(dreams)

    def test_round_trip(self) -> None:
        from dreambank.parsing.html import ParsedArchive

        series = SeriesFrame.build(
            pd.DataFrame(
                {
                    "series_id": ["ringo"],
                    "language": ["en"],
                    "series_title": ["Ringo"],
                    "reported_count": [16],
                    "series_years_raw": ["1964"],
                    "dreamer_sex_raw": ["male"],
                    "parsed_count": [1],
                }
            )
        )
        dreams = DreamsFrame.build(
            pd.DataFrame(
                {
                    "dream_id": ["ringo:1"],
                    "series_id": ["ringo"],
                    "dream_number_raw": ["1"],
                    "header_metadata_raw": ["10/07/64"],
                    "text": ["A dream."],
                    "word_count_reported": [2],
                    "language": ["en"],
                    "dreamer_sex_raw": ["male"],
                    "series_years_raw": ["1964"],
                    "source_file": ["ringo/dreams.html"],
                }
            )
        )
        parsed = ParsedArchive(series=series, dreams=dreams, unique_dreams=dreams)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_tables(parsed, folder)
            restored = read_archive(folder)
        self.assertIs(type(restored.series), SeriesFrame)
        self.assertIs(type(restored.dreams), DreamsFrame)
        pd.testing.assert_frame_equal(restored.series, series, check_exact=True)
        pd.testing.assert_frame_equal(restored.dreams, dreams, check_exact=True)


class ScrapeIdTests(unittest.TestCase):
    def test_catalog_ids_and_unsafe_values(self) -> None:
        ids = series_ids_from_html((HTML / "grid.html").read_bytes())
        self.assertEqual(len(ids), 96)
        self.assertIn("ringo", ids)
        self.assertIn("vonuslar.de", ids)
        sample = """
        <input type=checkbox name=series value="alta">
        <input type=checkbox name=series value="../evil">
        <input type=checkbox name=series value="/tmp/x">
        <input type=checkbox name=series value="C:/Windows">
        <input type=checkbox name=series value="alta">
        <input value="second" name=series>
        <input type=checkbox name=series value="foo/bar">
        """.encode()
        self.assertEqual(series_ids_from_html(sample), ["alta", "second"])
        for unsafe in ("../evil", "/tmp/x", "C:/Windows", "foo/bar", "..", ""):
            self.assertFalse(is_safe_series_id(unsafe))

    def test_unsafe_id_is_rejected_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsafe series id"):
                from dreambank import scrape

                scrape(ScrapeConfig(output=Path(tmp), series_ids=("../evil",)))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_files_are_not_redownloaded(self) -> None:
        from dreambank import scrape

        report = scrape(ScrapeConfig(output=HTML, refresh=False, series_ids=("ringo",)))
        self.assertEqual(list(report["series_id"]), ["ringo"])
        self.assertFalse(bool(report["downloaded"].iloc[0]))
        ringo = (HTML / "ringo").resolve()
        self.assertEqual(report["info_path"].iloc[0], (ringo / "info.html").as_posix())
        self.assertEqual(report["dreams_path"].iloc[0], (ringo / "dreams.html").as_posix())
        self.assertEqual(report["moreinfo_path"].iloc[0], (ringo / "moreinfo.html").as_posix())


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive = read_archive(PROCESSED)

    def test_read_parquet_returns_subclasses(self) -> None:
        self.assertIs(type(self.archive.series), SeriesFrame)
        self.assertIs(type(self.archive.dreams), DreamsFrame)
        self.assertIs(type(self.archive.unique_dreams), DreamsFrame)
        mismatch = set(
            self.archive.series.loc[
                self.archive.series["reported_count"] != self.archive.series["parsed_count"],
                "series_id",
            ]
        )
        self.assertEqual(mismatch, {"b", "college-f", "kenneth", "vonuslar.de"})

    def test_parse_matches_parquet(self) -> None:
        parsed = parse_html(HTML)
        self.assertEqual(len(parsed.series), 96)
        self.assertEqual(len(parsed.dreams), 45058)
        self.assertEqual(len(parsed.unique_dreams), 34838)
        pd.testing.assert_frame_equal(parsed.series, self.archive.series, check_exact=True)
        pd.testing.assert_frame_equal(parsed.dreams, self.archive.dreams, check_exact=True)
        pd.testing.assert_frame_equal(parsed.unique_dreams, self.archive.unique_dreams, check_exact=True)


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive = read_archive(PROCESSED)

    def test_emotions_and_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            emotions = IzzyEmotionsByAge(IzzyAgeConfig(figure_path=folder / "emotions.png"))
            emotions.prepare(self.archive)
            emotions.analyze()
            assert_csv(emotions.get_results(), RESULTS / "izzy_by_age.csv")
            emotions.plot()
            self.assertGreater((folder / "emotions.png").stat().st_size, 1000)
            emotions.to_csv(folder / "emotions.csv")
            plt.close("all")

            length = IzzyLengthVocabulary(IzzyAgeConfig(figure_path=folder / "length.png"))
            length.prepare(self.archive)
            length.analyze()
            got = pd.DataFrame(length.get_results())
            expected = pd.read_csv(RESULTS / "izzy_by_age.csv", encoding="utf-8-sig")
            pd.testing.assert_frame_equal(
                got,
                expected[["age_group", "token_count", "mattr_50"]],
                check_dtype=False,
                rtol=1e-8,
                atol=1e-8,
            )
            length.plot()
            self.assertGreater((folder / "length.png").stat().st_size, 1000)
            plt.close("all")

    def test_topics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            topics = IzzyTopics(IzzyTopicsConfig(figure_path=folder / "topics.png"))
            topics.prepare(self.archive)
            results = topics.analyze()
            assert_csv(results.by_age, RESULTS / "topics_by_age.csv", rtol=1e-5, atol=1e-6)
            assert_csv(
                results.terms,
                RESULTS / "topic_terms.csv",
                rename={"Компонент": "component", "Ключевые слова": "terms"},
            )
            topics.plot()
            topics.to_csv(folder / "topics.csv")
            self.assertGreater((folder / "topics.png").stat().st_size, 1000)
            plt.close("all")

    def test_gender_and_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            gender = GenderComparison(GenderConfig(figure_path=folder / "gender.png"))
            gender.prepare(self.archive)
            gender.analyze()
            assert_csv(gender.get_results(), RESULTS / "gender_comparison.csv")
            gender.plot()
            self.assertGreater((folder / "gender.png").stat().st_size, 1000)

            words = WordsByGender(WordsByGenderConfig(figure_path=folder / "words.png"))
            words.prepare(self.archive)
            words.analyze()
            assert_csv(words.get_results(), RESULTS / "words_by_gender.csv")
            words.plot()
            self.assertGreater((folder / "words.png").stat().st_size, 1000)
            plt.close("all")


class PipelineTests(unittest.TestCase):
    def test_default_output_is_a_spb_timestamp_under_package_results(self) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from dreambank.pipeline import SPB, default_output_dir, results_root

        path = default_output_dir()
        self.assertEqual(path.parent, results_root())
        self.assertEqual(results_root().name, "results")
        self.assertEqual(results_root().parent.name, "dreambank")
        parsed = datetime.strptime(path.name, "%d_%m_%Y_%H_%M").replace(tzinfo=ZoneInfo("Europe/Moscow"))
        self.assertLess(abs((datetime.now(SPB) - parsed).total_seconds()), 120)

    def test_run_writes_the_notebook_outputs(self) -> None:
        from dreambank.pipeline import PipelineConfig, run

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertLogs(level="INFO") as captured:
                archive = run(PipelineConfig(html_root=HTML, output=output))
            text = "\n".join(captured.output)
            self.assertIn("Собрал информацию", text)
            self.assertIn("Обработал архив", text)
            for experiment in (
                "IzzyEmotionsByAge",
                "IzzyLengthVocabulary",
                "IzzyTopics",
                "GenderComparison",
                "WordsByGender",
            ):
                self.assertIn(f"Запустил эксперимент {experiment}", text)
                self.assertIn(f"Закончил эксперимент {experiment}", text)
            self.assertEqual(len(archive.series), 96)
            self.assertEqual(len(archive.dreams), 45058)
            self.assertEqual(len(archive.unique_dreams), 34838)
            assert_csv(pd.read_csv(output / "izzy_by_age.csv", encoding="utf-8-sig"), RESULTS / "izzy_by_age.csv")
            assert_csv(
                pd.read_csv(output / "topics_by_age.csv", encoding="utf-8-sig"),
                RESULTS / "topics_by_age.csv",
                rtol=1e-5,
                atol=1e-6,
            )
            assert_csv(
                pd.read_csv(output / "topic_terms.csv", encoding="utf-8-sig"),
                RESULTS / "topic_terms.csv",
                rename={"Компонент": "component", "Ключевые слова": "terms"},
            )
            assert_csv(
                pd.read_csv(output / "gender_comparison.csv", encoding="utf-8-sig"),
                RESULTS / "gender_comparison.csv",
            )
            assert_csv(
                pd.read_csv(output / "words_by_gender.csv", encoding="utf-8-sig"),
                RESULTS / "words_by_gender.csv",
            )
            for name in (
                "01_emotions_by_age.png",
                "02_length_vocabulary.png",
                "03_topics_by_age.png",
                "04_gender_metrics.png",
                "05_words_by_gender.png",
            ):
                figure = output / "figures" / name
                self.assertGreater(figure.stat().st_size, 1000)
            self.assertTrue((output / "processed" / "dreams_simple.parquet").is_file())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
