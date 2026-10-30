from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from dreambank.analytics.base import Analysis, pyplot
from dreambank.analytics.config import IzzyAgeConfig, IzzyTopicsConfig
from dreambank.analytics.constants import EMOTIONS
from dreambank.analytics.features import mean_by_age, prepare_age_features
from dreambank.analytics.frames import (
    FEATURE_COLUMNS,
    AgeDreamFeaturesFrame,
    EmotionsByAgeFrame,
    LengthVocabularyFrame,
    TopicDreamFrame,
    TopicResults,
    TopicTermsFrame,
    TopicsByAgeFrame,
)
from dreambank.common.frame import column_schema
from dreambank.parsing.html import ParsedArchive

AgeResultT = TypeVar("AgeResultT", bound=pd.DataFrame)


class _IzzyAgeAnalysis(Analysis[AgeResultT]):
    def __init__(self, config: IzzyAgeConfig) -> None:
        super().__init__(config)
        self.config: IzzyAgeConfig = config

    def prepare(self, archive: ParsedArchive) -> AgeDreamFeaturesFrame:
        self._results = None
        self._working = prepare_age_features(archive, self.config.ages)
        return self._working

    def _age_table(self, columns: list[str]) -> pd.DataFrame:
        working = self._require_working()
        if not isinstance(working, AgeDreamFeaturesFrame):
            raise RuntimeError("prepare() did not build an age frame")
        labels = [band.label for band in self.config.ages]
        return mean_by_age(working, columns, labels)


class IzzyEmotionsByAge(_IzzyAgeAnalysis[EmotionsByAgeFrame]):
    """Mean NRC rates and VADER compound for the Izzy age series."""

    def analyze(self) -> EmotionsByAgeFrame:
        table = self._age_table(list(FEATURE_COLUMNS))
        self._results = EmotionsByAgeFrame.build(table)
        return self._results

    def plot(self) -> Figure:
        plt = pyplot()
        frame = pd.DataFrame(self.get_results())
        ages = frame["age_group"].tolist()
        emotions = list(EMOTIONS)
        heat = frame[emotions].to_numpy().T
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1]})
        image = axes[0].imshow(heat, aspect="auto", cmap="YlOrRd")
        axes[0].set_xticks(range(len(ages)), ages)
        axes[0].set_yticks(range(len(emotions)), emotions)
        axes[0].set_title("Izzy: эмоциональная лексика по возрасту")
        for row in range(len(emotions)):
            for col in range(len(ages)):
                axes[0].text(col, row, f"{heat[row, col]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axes[0], label="Средняя частота на 1000 токенов сна")
        axes[1].plot(ages, frame["vader"], marker="o")
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[1].set(xlabel="Возраст", ylabel="VADER compound", title="Средняя тональность")
        fig.tight_layout()
        self._save(fig)
        return fig


class IzzyLengthVocabulary(_IzzyAgeAnalysis[LengthVocabularyFrame]):
    """Mean token count and MATTR-50 for the Izzy age series."""

    def analyze(self) -> LengthVocabularyFrame:
        table = self._age_table(["token_count", "mattr_50"])
        self._results = LengthVocabularyFrame.build(table)
        return self._results

    def plot(self) -> Figure:
        plt = pyplot()
        frame = pd.DataFrame(self.get_results())
        ages = frame["age_group"].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ages, frame["token_count"], marker="o")
        axes[0].set(title="Средняя длина записи", xlabel="Возраст", ylabel="Токенов")
        axes[1].plot(ages, frame["mattr_50"], marker="o", color="darkorange")
        axes[1].set(title="Разнообразие словаря, MATTR-50", xlabel="Возраст", ylabel="Доля разных слов")
        fig.tight_layout()
        self._save(fig)
        return fig


class IzzyTopics(Analysis[TopicResults]):
    """TF-IDF and 8-component NMF on the Izzy age series."""

    def __init__(self, config: IzzyTopicsConfig) -> None:
        super().__init__(config)
        self.config: IzzyTopicsConfig = config

    def prepare(self, archive: ParsedArchive) -> TopicDreamFrame:
        self._results = None
        plain = pd.DataFrame(archive.dreams)
        ids = [band.series_id for band in self.config.ages]
        subset = plain.loc[plain["series_id"].isin(ids)].reset_index(drop=True)
        present = set(subset["series_id"])
        missing = [series_id for series_id in ids if series_id not in present]
        if missing:
            raise ValueError(f"No dreams for series: {', '.join(missing)}")
        labels = {band.series_id: band.label for band in self.config.ages}
        subset = subset.sort_values(["series_id", "dream_number_raw"]).reset_index(drop=True)
        subset["age_group"] = subset["series_id"].map(labels)
        self._working = TopicDreamFrame.build(
            subset[["series_id", "dream_number_raw", "text", "age_group"]]
        )
        return self._working

    def analyze(self) -> TopicResults:
        working = self._require_working()
        if not isinstance(working, TopicDreamFrame):
            raise RuntimeError("prepare() did not build a topic frame")
        expected = tuple(
            spec.name for spec in column_schema(TopicsByAgeFrame) if spec.name != "age_group"
        )
        if self.config.labels != expected:
            raise ValueError("Topic labels must match TopicsByAgeFrame columns")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features,
            sublinear_tf=self.config.sublinear_tf,
        )
        matrix = vectorizer.fit_transform(working["text"])
        model = NMF(
            n_components=self.config.n_components,
            init="nndsvda",
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )
        weights = model.fit_transform(matrix)
        shares = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        vocabulary = vectorizer.get_feature_names_out()
        terms = TopicTermsFrame.build(
            pd.DataFrame(
                {
                    "component": list(self.config.labels),
                    "terms": [
                        ", ".join(vocabulary[component.argsort()[-self.config.top_terms :][::-1]])
                        for component in model.components_
                    ],
                }
            )
        )
        age_labels = [band.label for band in self.config.ages]
        topic_table = pd.DataFrame(shares, columns=list(self.config.labels))
        topic_table["age_group"] = pd.Categorical(
            working["age_group"].to_numpy(),
            categories=age_labels,
            ordered=True,
        )
        means = (
            topic_table.groupby("age_group", observed=True)[list(self.config.labels)]
            .mean()
            .reindex(age_labels)
        )
        by_age = means.reset_index()
        by_age["age_group"] = by_age["age_group"].astype(str)
        self._results = TopicResults(by_age=TopicsByAgeFrame.build(by_age), terms=terms)
        return self._results

    def plot(self) -> Figure:
        plt = pyplot()
        frame = pd.DataFrame(self.get_results().by_age)
        ages = frame["age_group"].tolist()
        labels = [column for column in frame.columns if column != "age_group"]
        heat = frame[labels].to_numpy().T
        fig, ax = plt.subplots(figsize=(11, 6))
        image = ax.imshow(heat, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(ages)), ages)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Возраст", title="Izzy: средний нормированный вес компонента")
        for row in range(len(labels)):
            for col in range(len(ages)):
                ax.text(col, row, f"{heat[row, col]:.0%}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Нормированный вес")
        fig.tight_layout()
        self._save(fig)
        return fig

    def to_csv(self, path: str | Path) -> None:
        self.get_results().by_age.to_csv(path, index=False, encoding="utf-8-sig")
