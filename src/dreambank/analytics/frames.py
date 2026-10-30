from dataclasses import dataclass

from dreambank.analytics.constants import EMOTIONS, TOPIC_LABELS
from dreambank.common.frame import TypedFrame
from dreambank.parsing.frames import DreamsFrame


class DreamFeaturesFrame(DreamsFrame):
    token_count: int
    mattr_50: float
    vader: float
    anger: float
    fear: float
    anticipation: float
    trust: float
    surprise: float
    sadness: float
    disgust: float
    joy: float


class AgeDreamFeaturesFrame(DreamFeaturesFrame):
    age_group: str


class EmotionsByAgeFrame(TypedFrame):
    age_group: str
    token_count: float
    mattr_50: float
    vader: float
    anger: float
    fear: float
    anticipation: float
    trust: float
    surprise: float
    sadness: float
    disgust: float
    joy: float


class LengthVocabularyFrame(TypedFrame):
    age_group: str
    token_count: float
    mattr_50: float


class TopicsByAgeFrame(TypedFrame):
    age_group: str
    Диалоги: float
    Школа: float
    Воспоминание: float
    Фильмы: float


TopicsByAgeFrame.__annotations__ = {
    "age_group": str,
    **{label: float for label in TOPIC_LABELS},
}


class TopicTermsFrame(TypedFrame):
    component: str
    terms: str


@dataclass(frozen=True)
class TopicResults:
    by_age: TopicsByAgeFrame
    terms: TopicTermsFrame


class GenderComparisonFrame(TypedFrame):
    metric: str
    female_mean: float
    male_mean: float
    p: float
    cliffs_delta: float
    p_adjusted: float


class WordsByGenderFrame(TypedFrame):
    word: str
    female_count: int
    male_count: int
    female: float
    male: float
    shared: float
    difference: float


class TopicDreamFrame(TypedFrame):
    series_id: str
    dream_number_raw: str
    text: str
    age_group: str


class GenderTextFrame(TypedFrame):
    series_id: str
    text: str


FEATURE_COLUMNS = ("token_count", "mattr_50", "vader", *EMOTIONS)
