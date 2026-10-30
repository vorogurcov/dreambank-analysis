import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu

from dreambank.analytics.base import Analysis, pyplot
from dreambank.analytics.config import GenderConfig
from dreambank.analytics.constants import EMOTIONS, FEMALE_COLOR, MALE_COLOR
from dreambank.analytics.features import attach_features
from dreambank.analytics.frames import DreamFeaturesFrame, GenderComparisonFrame
from dreambank.parsing.frames import DreamsFrame
from dreambank.parsing.html import ParsedArchive

METRICS = ("token_count", "mattr_50", "vader", *EMOTIONS)


class GenderComparison(Analysis[GenderComparisonFrame]):
    """Mann-Whitney comparison of college-f and college-m, with Benjamini-Hochberg adjustment."""

    def __init__(self, config: GenderConfig) -> None:
        super().__init__(config)
        self.config: GenderConfig = config

    def prepare(self, archive: ParsedArchive) -> DreamFeaturesFrame:
        self._results = None
        plain = pd.DataFrame(archive.dreams)
        ids = [self.config.female.series_id, self.config.male.series_id]
        subset = plain.loc[plain["series_id"].isin(ids)].reset_index(drop=True)
        present = set(subset["series_id"])
        missing = [series_id for series_id in ids if series_id not in present]
        if missing:
            raise ValueError(f"No dreams for series: {', '.join(missing)}")
        self._working = attach_features(DreamsFrame.build(subset))
        return self._working

    def analyze(self) -> GenderComparisonFrame:
        working = self._require_working()
        if not isinstance(working, DreamFeaturesFrame):
            raise RuntimeError("prepare() did not build a feature frame")
        plain = pd.DataFrame(working)
        rows: list[dict[str, float | str]] = []
        for metric in METRICS:
            female = plain.loc[plain["series_id"] == self.config.female.series_id, metric].dropna()
            male = plain.loc[plain["series_id"] == self.config.male.series_id, metric].dropna()
            if female.empty or male.empty:
                raise ValueError(f"No observations for {metric}")
            test = mannwhitneyu(female, male, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "female_mean": float(female.mean()),
                    "male_mean": float(male.mean()),
                    "p": float(test.pvalue),
                    "cliffs_delta": 2 * float(test.statistic) / (len(female) * len(male)) - 1,
                }
            )
        comparison = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
        adjusted = comparison["p"].to_numpy() * len(METRICS) / np.arange(1, len(METRICS) + 1)
        comparison["p_adjusted"] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
        self._results = GenderComparisonFrame.build(comparison)
        return self._results

    def plot(self) -> Figure:
        plt = pyplot()
        results = pd.DataFrame(self.get_results()).set_index("metric")
        labels = [self.config.female.label, self.config.male.label]
        colors = [FEMALE_COLOR, MALE_COLOR]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        token = results.loc["token_count", ["female_mean", "male_mean"]]
        token.index = labels
        token.plot.bar(ax=axes[0], color=colors, rot=0)
        axes[0].set(title="Средняя длина записи", ylabel="Токенов", xlabel="")
        emotions = results.loc[list(EMOTIONS), ["female_mean", "male_mean"]]
        emotions.columns = labels
        emotions.plot.bar(ax=axes[1], color=colors, rot=45)
        axes[1].set(title="Эмоциональная лексика", ylabel="Средняя частота на 1000 токенов", xlabel="")
        fig.tight_layout()
        self._save(fig)
        return fig
