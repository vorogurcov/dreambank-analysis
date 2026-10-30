from typing import Literal

from dreambank.common.frame import TypedFrame


class SeriesFrame(TypedFrame):
    series_id: str
    language: Literal["en", "de"]
    series_title: str
    reported_count: int
    series_years_raw: str
    dreamer_sex_raw: str
    parsed_count: int


class DreamsFrame(TypedFrame):
    dream_id: str
    series_id: str
    dream_number_raw: str
    header_metadata_raw: str
    text: str
    word_count_reported: int
    language: Literal["en", "de"]
    dreamer_sex_raw: str
    series_years_raw: str
    source_file: str
