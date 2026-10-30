from dreambank.common.frame import TypedFrame


class ScrapeReportFrame(TypedFrame):
    series_id: str
    info_path: str
    dreams_path: str
    moreinfo_path: str
    downloaded: bool
