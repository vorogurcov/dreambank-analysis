"""Download DreamBank series pages into the snapshot layout."""

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import pandas as pd

from dreambank.scraping.frames import ScrapeReportFrame

GRID_URL = "https://dreambank.net/grid.cgi"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_INVALID_FILENAME = re.compile(r'[<>:"|?*\x00-\x1f]')
_INPUT_TAG = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_PAGES = (
    ("info.html", "https://dreambank.net/more_info.cgi?series={series}"),
    ("dreams.html", "https://dreambank.net/random_sample.cgi?series={series}"),
    ("moreinfo.html", "https://dreambank.net/more_info.cgi?series={series}&further=1"),
)


def default_html_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "notebooks").is_dir():
            return parent / "notebooks" / "data" / "raw" / "html"
    return Path("notebooks/data/raw/html")


@dataclass(frozen=True)
class ScrapeConfig:
    """Where and how to download series pages.

    ``refresh=False`` leaves an existing file in place. ``series_ids`` limits the
    download to those catalog ids; ``None`` means every id on ``grid.cgi``.
    """

    output: Path = field(default_factory=default_html_root)
    refresh: bool = False
    timeout: float = 60.0
    pause_seconds: float = 1.0
    retries: int = 3
    user_agent: str = "DreamBank educational project"
    series_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.pause_seconds < 0:
            raise ValueError("pause_seconds must be zero or positive")
        if self.retries < 1:
            raise ValueError("retries must be at least 1")
        if self.series_ids is None:
            return
        if isinstance(self.series_ids, str):
            raise TypeError("series_ids must be a sequence of ids, not a string")
        series_ids = tuple(self.series_ids)
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("series_ids contains duplicates")
        object.__setattr__(self, "series_ids", series_ids)


def is_safe_series_id(series_id: str) -> bool:
    """Reject absolute ids, path separators, and ``..`` segments."""
    if not series_id or series_id != series_id.strip():
        return False
    if _INVALID_FILENAME.search(series_id):
        return False
    path = Path(series_id)
    if path.is_absolute() or path.name != series_id:
        return False
    if series_id in {".", ".."} or ".." in path.parts:
        return False
    return True


def series_ids_from_html(payload: bytes) -> list[str]:
    """Ids from ``grid.cgi`` checkboxes, in page order, skipping unsafe values."""
    text = payload.decode("latin-1")
    found: list[str] = []
    seen: set[str] = set()
    for tag in _INPUT_TAG.findall(text):
        if _attribute(tag, "name") != "series":
            continue
        value = _attribute(tag, "value")
        if value is None or value in seen or not is_safe_series_id(value):
            continue
        seen.add(value)
        found.append(value)
    return found


def scrape(config: ScrapeConfig | None = None) -> ScrapeReportFrame:
    """Download ``grid.html`` and each series' info, dreams, and moreinfo pages."""
    settings = config if config is not None else ScrapeConfig()
    if settings.series_ids is not None:
        unsafe = [series_id for series_id in settings.series_ids if not is_safe_series_id(series_id)]
        if unsafe:
            raise ValueError(f"Unsafe series id: {unsafe[0]}")

    output = settings.output
    output.mkdir(parents=True, exist_ok=True)
    client = _Client(settings)
    grid_path = output / "grid.html"
    client.materialize(grid_path, GRID_URL)
    selected = series_ids_from_html(grid_path.read_bytes())
    if settings.series_ids is not None:
        known = set(selected)
        missing = [series_id for series_id in settings.series_ids if series_id not in known]
        if missing:
            raise ValueError(f"Series id is not in the catalog: {missing[0]}")
        selected = list(settings.series_ids)

    rows: list[dict[str, object]] = []
    root = output.resolve()
    for series_id in selected:
        folder = (output / series_id).resolve()
        if folder.parent != root:
            continue
        downloaded = False
        paths: dict[str, str] = {}
        quoted = quote(series_id, safe="")
        for filename, template in _PAGES:
            path = folder / filename
            downloaded = client.materialize(path, template.format(series=quoted)) or downloaded
            paths[filename] = path.resolve().as_posix()
        rows.append(
            {
                "series_id": series_id,
                "info_path": paths["info.html"],
                "dreams_path": paths["dreams.html"],
                "moreinfo_path": paths["moreinfo.html"],
                "downloaded": downloaded,
            }
        )
    if not rows:
        return ScrapeReportFrame.blank()
    return ScrapeReportFrame.build(pd.DataFrame(rows))


def _attribute(tag: str, name: str) -> str | None:
    match = re.search(
        rf"(?:^|\s){name}\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))",
        tag,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


class _Client:
    def __init__(self, config: ScrapeConfig) -> None:
        self.config = config
        self._requests = 0

    def materialize(self, path: Path, url: str) -> bool:
        if path.is_file() and not self.config.refresh:
            return False
        if self._requests and self.config.pause_seconds:
            time.sleep(self.config.pause_seconds)
        payload = self._fetch(url)
        self._requests += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
        return True

    def _fetch(self, url: str) -> bytes:
        for attempt in range(self.config.retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": self.config.user_agent},
                )
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                retryable = error.code in _RETRY_STATUSES
                if not retryable or attempt + 1 == self.config.retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt + 1 == self.config.retries:
                    raise
            time.sleep(self.config.pause_seconds)
        raise RuntimeError(f"Failed to download {url}")
