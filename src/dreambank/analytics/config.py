from dataclasses import dataclass
from pathlib import Path

from dreambank.analytics.constants import EXTRA_STOP_WORDS, TOPIC_LABELS


@dataclass(frozen=True)
class AgeBand:
    series_id: str
    label: str


@dataclass(frozen=True)
class SeriesGroup:
    series_id: str
    label: str


DEFAULT_AGES = (
    AgeBand("izzy12", "12"),
    AgeBand("izzy13", "13"),
    AgeBand("izzy14", "14"),
    AgeBand("izzy15", "15"),
    AgeBand("izzy16", "16"),
    AgeBand("izzy17", "17"),
    AgeBand("izzy18_21", "18\u201321"),
    AgeBand("izzy22_25", "22\u201325"),
)

FEMALE = SeriesGroup("college-f", "Женщины")
MALE = SeriesGroup("college-m", "Мужчины")


@dataclass(frozen=True)
class IzzyAgeConfig:
    ages: tuple[AgeBand, ...] = DEFAULT_AGES
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        _unique_ages(self.ages)


@dataclass(frozen=True)
class IzzyTopicsConfig:
    ages: tuple[AgeBand, ...] = DEFAULT_AGES
    labels: tuple[str, ...] = TOPIC_LABELS
    min_df: int = 10
    max_df: float = 0.85
    max_features: int = 6000
    ngram_range: tuple[int, int] = (1, 2)
    n_components: int = 8
    random_state: int = 42
    max_iter: int = 700
    sublinear_tf: bool = True
    top_terms: int = 12
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        _unique_ages(self.ages)
        if len(self.labels) != self.n_components:
            raise ValueError("labels and n_components must have the same length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("topic labels must be unique")


@dataclass(frozen=True)
class GenderConfig:
    female: SeriesGroup = FEMALE
    male: SeriesGroup = MALE
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        if self.female.series_id == self.male.series_id:
            raise ValueError("female and male series must differ")


@dataclass(frozen=True)
class WordsByGenderConfig:
    female: SeriesGroup = FEMALE
    male: SeriesGroup = MALE
    extra_stop_words: tuple[str, ...] = EXTRA_STOP_WORDS
    min_count: int = 10
    per: int = 10_000
    min_length: int = 3
    top_shared: int = 15
    top_difference: int = 12
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        if self.female.series_id == self.male.series_id:
            raise ValueError("female and male series must differ")


def _unique_ages(ages: tuple[AgeBand, ...]) -> None:
    ids = [band.series_id for band in ages]
    labels = [band.label for band in ages]
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise ValueError("Age bands need unique series ids and labels")
