from pathlib import Path
from typing import TypeVar

import pandas as pd

from dreambank.common.frame import TypedFrame
from dreambank.parsing.frames import DreamsFrame, SeriesFrame
from dreambank.parsing.html import ParsedArchive

FrameT = TypeVar("FrameT", bound=TypedFrame)

_TABLES = (
    ("series_simple", "series", SeriesFrame),
    ("dreams_simple", "dreams", DreamsFrame),
    ("dreams_unique_simple", "unique_dreams", DreamsFrame),
)


def write_tables(archive: ParsedArchive, directory: Path | str) -> None:
    """Write csv (``utf-8-sig``) and parquet for the three archive tables."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    for name, attribute, _frame in _TABLES:
        table = getattr(archive, attribute)
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
        table.to_parquet(folder / f"{name}.parquet", index=False)


def read_parquet(path: Path | str, frame: type[FrameT]) -> FrameT:
    """Read a parquet file back into the annotated frame subclass."""
    return frame.build(pd.read_parquet(path))


def read_archive(directory: Path | str) -> ParsedArchive:
    """Read ``series_simple``, ``dreams_simple``, and ``dreams_unique_simple``."""
    folder = Path(directory)
    return ParsedArchive(
        series=read_parquet(folder / "series_simple.parquet", SeriesFrame),
        dreams=read_parquet(folder / "dreams_simple.parquet", DreamsFrame),
        unique_dreams=read_parquet(folder / "dreams_unique_simple.parquet", DreamsFrame),
    )
