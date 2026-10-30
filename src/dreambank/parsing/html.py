"""Read DreamBank ``info.html`` and ``dreams.html`` into typed frames."""

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from bs4 import BeautifulSoup

from dreambank.parsing.frames import DreamsFrame, SeriesFrame

_WHITESPACE = re.compile(r"\s+")
_HEADER = re.compile(r"^#(\S+)(?:\s+\((.*)\))?$")
_WORD_COUNT = re.compile(r"\(([\d,]+)\s+words?\)\s*$", re.IGNORECASE)
_SERIES_LABELS = {
    "Dream series:": "series_title",
    "Number of dreams:": "reported_count",
    "Year:": "series_years_raw",
    "Sex of the dreamer(s):": "dreamer_sex_raw",
}


@dataclass
class ParsedArchive:
    series: SeriesFrame
    dreams: DreamsFrame
    unique_dreams: DreamsFrame


def clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def read_series(folder: Path) -> dict[str, object]:
    soup = _soup(folder / "info.html")
    row: dict[str, object] = {
        "series_id": folder.name,
        "language": "de" if folder.name.endswith(".de") else "en",
    }
    for bold in soup.find_all("b"):
        label = clean(bold.get_text(" ", strip=True))
        if label not in _SERIES_LABELS:
            continue
        pieces: list[str] = []
        for sibling in bold.next_siblings:
            if getattr(sibling, "name", None) == "br":
                break
            if getattr(sibling, "name", None):
                pieces.append(sibling.get_text(" ", strip=True))
            else:
                pieces.append(str(sibling))
        row[_SERIES_LABELS[label]] = clean(" ".join(pieces))
    try:
        reported = row["reported_count"]
    except KeyError as error:
        raise ValueError(f"{folder.name}: missing series field") from error
    row["reported_count"] = int(str(reported).replace(",", ""))
    return row


def read_dreams(folder: Path, series: dict[str, object], html_root: Path) -> list[dict[str, object]]:
    soup = _soup(folder / "dreams.html")
    source = source_file(folder / "dreams.html", html_root)
    rows: list[dict[str, object]] = []
    for span in soup.find_all("span"):
        header_parts: list[str] = []
        for child in span.contents:
            if getattr(child, "name", None) == "br":
                break
            if getattr(child, "name", None):
                header_parts.append(child.get_text(" ", strip=True))
            else:
                header_parts.append(str(child))
        header = clean(" ".join(header_parts))
        if not header.startswith("#"):
            continue
        match = _HEADER.match(header)
        if match is None:
            raise ValueError(f"{folder.name}: unrecognized dream header {header!r}")
        number, metadata = match.groups()
        body = clean(span.get_text(" ", strip=True))[len(header) :].strip()
        word_count = _WORD_COUNT.search(body)
        if word_count is None:
            raise ValueError(f"{folder.name}: dream #{number} has no word count")
        text = clean(body[: word_count.start()])
        rows.append(
            {
                "dream_id": f"{folder.name}:{number}",
                "series_id": folder.name,
                "dream_number_raw": number,
                "header_metadata_raw": metadata or "",
                "text": text,
                "word_count_reported": int(word_count.group(1).replace(",", "")),
                "language": series["language"],
                "dreamer_sex_raw": series["dreamer_sex_raw"],
                "series_years_raw": series["series_years_raw"],
                "source_file": source,
            }
        )
    return rows


def source_file(path: Path, html_root: Path) -> str:
    """Dream path relative to the snapshot folder that contains ``data/raw/html``.

    Any other root is stored relative to ``html_root`` itself.
    """
    path = path.resolve()
    html_root = html_root.resolve()
    if html_root.name == "html" and html_root.parent.name == "raw" and html_root.parent.parent.name == "data":
        anchor = html_root.parent.parent.parent
    else:
        anchor = html_root
    return path.relative_to(anchor).as_posix()


def parse_html(root: Path | str) -> ParsedArchive:
    """Parse every series folder under ``root`` that has ``info.html``."""
    html_root = Path(root)
    series_rows: list[dict[str, object]] = []
    dream_rows: list[dict[str, object]] = []
    folders = sorted((path for path in html_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    for folder in folders:
        if not (folder / "info.html").is_file():
            continue
        series = read_series(folder)
        records = read_dreams(folder, series, html_root)
        series["parsed_count"] = len(records)
        series_rows.append(series)
        dream_rows.extend(records)

    series_frame = SeriesFrame.build(pd.DataFrame(series_rows)) if series_rows else SeriesFrame.blank()
    dreams = DreamsFrame.build(pd.DataFrame(dream_rows)) if dream_rows else DreamsFrame.blank()
    _reject_bad_dreams(dreams)
    unique = DreamsFrame.build(pd.DataFrame(dreams).drop_duplicates("text").reset_index(drop=True))
    return ParsedArchive(series=series_frame, dreams=dreams, unique_dreams=unique)


def _soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_bytes(), "html.parser", from_encoding="windows-1252")


def _reject_bad_dreams(dreams: DreamsFrame) -> None:
    empty = dreams.loc[dreams["text"].str.len().eq(0), "dream_id"]
    if not empty.empty:
        raise ValueError(f"Empty dream text: {empty.iloc[0]}")
    if not dreams["dream_id"].is_unique:
        repeated = dreams.loc[dreams["dream_id"].duplicated(), "dream_id"].iloc[0]
        raise ValueError(f"Repeated dream_id: {repeated}")
