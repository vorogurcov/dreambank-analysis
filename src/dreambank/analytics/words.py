from collections import Counter

import pandas as pd
from matplotlib.figure import Figure
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from dreambank.analytics.base import Analysis, pyplot
from dreambank.analytics.config import WordsByGenderConfig
from dreambank.analytics.constants import FEMALE_COLOR, MALE_COLOR
from dreambank.analytics.features import tokens
from dreambank.analytics.frames import GenderTextFrame, WordsByGenderFrame
from dreambank.parsing.html import ParsedArchive


class WordsByGender(Analysis[WordsByGenderFrame]):
    """Stemmed content-word rates per 10,000 tokens for the two college series."""

    def __init__(self, config: WordsByGenderConfig) -> None:
        super().__init__(config)
        self.config: WordsByGenderConfig = config

    def prepare(self, archive: ParsedArchive) -> GenderTextFrame:
        self._results = None
        plain = pd.DataFrame(archive.dreams)
        ids = [self.config.female.series_id, self.config.male.series_id]
        subset = plain.loc[plain["series_id"].isin(ids), ["series_id", "text"]].reset_index(drop=True)
        present = set(subset["series_id"])
        missing = [series_id for series_id in ids if series_id not in present]
        if missing:
            raise ValueError(f"No dreams for series: {', '.join(missing)}")
        self._working = GenderTextFrame.build(subset)
        return self._working

    def analyze(self) -> WordsByGenderFrame:
        working = self._require_working()
        if not isinstance(working, GenderTextFrame):
            raise RuntimeError("prepare() did not build a text frame")
        stop_words = set(ENGLISH_STOP_WORDS) | set(self.config.extra_stop_words)
        stemmer = SnowballStemmer("english")
        groups = (self.config.female, self.config.male)
        counts: dict[str, Counter[str]] = {}
        forms: dict[str, dict[str, Counter[str]]] = {}
        totals: dict[str, int] = {}
        for group in groups:
            counts[group.label] = Counter()
            forms[group.label] = {}
            texts = working.loc[working["series_id"] == group.series_id, "text"]
            for text in texts:
                for word in tokens(text):
                    if word in stop_words or len(word) < self.config.min_length:
                        continue
                    stem = stemmer.stem(word)
                    counts[group.label][stem] += 1
                    forms[group.label].setdefault(stem, Counter())[word] += 1
            totals[group.label] = sum(counts[group.label].values())
            if totals[group.label] == 0:
                raise ValueError(f"No content tokens for {group.label}")

        female_label = self.config.female.label
        male_label = self.config.male.label
        rows: list[dict[str, float | int | str]] = []
        for stem in sorted(set(counts[female_label]) | set(counts[male_label])):
            female_count = counts[female_label][stem]
            male_count = counts[male_label][stem]
            if female_count + male_count < self.config.min_count:
                continue
            spellings = forms[female_label].get(stem, Counter()) + forms[male_label].get(stem, Counter())
            female_rate = female_count * self.config.per / totals[female_label]
            male_rate = male_count * self.config.per / totals[male_label]
            rows.append(
                {
                    "word": spellings.most_common(1)[0][0],
                    "female_count": female_count,
                    "male_count": male_count,
                    "female": female_rate,
                    "male": male_rate,
                    "shared": min(female_rate, male_rate),
                    "difference": female_rate - male_rate,
                }
            )
        self._results = WordsByGenderFrame.build(pd.DataFrame(rows))
        return self._results

    def plot(self) -> Figure:
        plt = pyplot()
        table = pd.DataFrame(self.get_results())
        common = table.nlargest(self.config.top_shared, "shared")
        female_words = table.nlargest(self.config.top_difference, "difference")
        male_words = table.nsmallest(self.config.top_difference, "difference")
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        common.set_index("word")[["female", "male"]].iloc[::-1].plot.barh(
            ax=axes[0],
            color=[FEMALE_COLOR, MALE_COLOR],
        )
        axes[0].legend([self.config.female.label, self.config.male.label])
        axes[0].set(title="Частые в обеих группах", xlabel="На 10 000 содержательных токенов", ylabel="")
        axes[1].barh(female_words["word"].iloc[::-1], female_words["difference"].iloc[::-1], color=FEMALE_COLOR)
        axes[1].set(title="Относительно чаще у женщин", xlabel="Разность частот: женщины − мужчины")
        axes[2].barh(male_words["word"].iloc[::-1], -male_words["difference"].iloc[::-1], color=MALE_COLOR)
        axes[2].set(title="Относительно чаще у мужчин", xlabel="Разность частот: мужчины − женщины")
        fig.tight_layout()
        self._save(fig)
        return fig