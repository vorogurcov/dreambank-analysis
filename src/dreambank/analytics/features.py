"""Token, VADER, NRC, and MATTR-50 features for one dream text."""

import re

import numpy as np
import pandas as pd
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from dreambank.analytics.config import AgeBand
from dreambank.analytics.constants import EMOTIONS
from dreambank.analytics.frames import AgeDreamFeaturesFrame, DreamFeaturesFrame
from dreambank.parsing.html import ParsedArchive
from dreambank.parsing.frames import DreamsFrame

_TOKEN = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FeatureEncoder:
    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._nrc = NRCLex()

    def encode(self, text: str) -> dict[str, float | int]:
        words = tokens(text)
        self._nrc.load_token_list(words)
        result: dict[str, float | int] = {
            "token_count": len(words),
            "vader": self._vader.polarity_scores(text)["compound"],
        }
        for emotion in EMOTIONS:
            result[emotion] = self._nrc.raw_emotion_scores.get(emotion, 0) * 1000 / max(len(words), 1)
        if len(words) >= 50:
            result["mattr_50"] = float(
                np.mean([len(set(words[i : i + 50])) / 50 for i in range(len(words) - 49)])
            )
        else:
            result["mattr_50"] = float("nan")
        return result


def attach_features(dreams: DreamsFrame) -> DreamFeaturesFrame:
    base = pd.DataFrame(dreams).reset_index(drop=True)
    encoder = FeatureEncoder()
    if base.empty:
        extra = pd.DataFrame(
            {
                "token_count": pd.Series(dtype="int64"),
                "mattr_50": pd.Series(dtype="float64"),
                "vader": pd.Series(dtype="float64"),
                **{emotion: pd.Series(dtype="float64") for emotion in EMOTIONS},
            }
        )
    else:
        extra = pd.DataFrame([encoder.encode(text) for text in base["text"]], index=base.index)
    return DreamFeaturesFrame.build(pd.concat([base, extra], axis=1))


def prepare_age_features(archive: ParsedArchive, ages: tuple[AgeBand, ...]) -> AgeDreamFeaturesFrame:
    plain = pd.DataFrame(archive.dreams)
    ids = [band.series_id for band in ages]
    subset = plain.loc[plain["series_id"].isin(ids)].reset_index(drop=True)
    present = set(subset["series_id"])
    missing = [series_id for series_id in ids if series_id not in present]
    if missing:
        raise ValueError(f"No dreams for series: {', '.join(missing)}")
    featured = pd.DataFrame(attach_features(DreamsFrame.build(subset)))
    labels = {band.series_id: band.label for band in ages}
    featured["age_group"] = featured["series_id"].map(labels)
    return AgeDreamFeaturesFrame.build(featured)


def mean_by_age(frame: AgeDreamFeaturesFrame, columns: list[str], labels: list[str]) -> pd.DataFrame:
    plain = pd.DataFrame(frame)
    plain["age_group"] = pd.Categorical(plain["age_group"], categories=labels, ordered=True)
    means = plain.groupby("age_group", observed=True)[columns].mean().reindex(labels)
    result = means.reset_index()
    result["age_group"] = result["age_group"].astype(str)
    return result
