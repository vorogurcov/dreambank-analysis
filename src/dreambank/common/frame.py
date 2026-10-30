"""DataFrame subclasses whose columns are declared on the class."""

from typing import Literal, Self, get_args, get_origin

import pandas as pd

ColumnType = type[str] | type[int] | type[float] | type[bool]


class FrameSchemaError(ValueError):
    """Raised when a frame's columns or dtypes do not match its annotations."""


class ColumnSpec:
    def __init__(
        self,
        name: str,
        scalar: ColumnType,
        allowed: frozenset[str] | None = None,
    ) -> None:
        self.name = name
        self.scalar = scalar
        self.allowed = allowed


def column_schema(cls: type) -> tuple[ColumnSpec, ...]:
    """Column specs in declaration order, including columns inherited from frames."""
    merged: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        if not base.__module__.startswith("dreambank"):
            continue
        for name, annotation in getattr(base, "__annotations__", {}).items():
            if name.startswith("_"):
                continue
            merged[name] = annotation
    return tuple(_spec(name, annotation) for name, annotation in merged.items())


def _spec(name: str, annotation: object) -> ColumnSpec:
    if get_origin(annotation) is Literal:
        allowed = get_args(annotation)
        if not allowed or not all(isinstance(value, str) for value in allowed):
            raise FrameSchemaError(f"{name} must be a Literal of strings")
        return ColumnSpec(name, str, frozenset(allowed))
    if annotation in (str, int, float, bool):
        return ColumnSpec(name, annotation)
    raise FrameSchemaError(f"Unsupported column annotation for {name}: {annotation!r}")


class TypedFrame(pd.DataFrame):
    """DataFrame built only through :meth:`build` or :meth:`blank`.

    Pandas operations may still return this subclass without checking the schema.
    Crossing a package boundary should go through :meth:`build`.
    """

    @classmethod
    def build(cls, data: object) -> Self:
        specs = column_schema(cls)
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        expected = [spec.name for spec in specs]
        actual = [str(column) for column in frame.columns]
        if len(actual) != len(set(actual)):
            raise FrameSchemaError(f"{cls.__name__} has duplicate columns")
        missing = [name for name in expected if name not in actual]
        extra = [name for name in actual if name not in expected]
        if missing or extra:
            raise FrameSchemaError(
                f"{cls.__name__} columns do not match the schema: "
                f"missing={missing}, extra={extra}"
            )
        ordered = frame.loc[:, expected].copy()
        for spec in specs:
            _check_column(cls.__name__, ordered[spec.name], spec)
        return cls(ordered)

    @classmethod
    def blank(cls) -> Self:
        specs = column_schema(cls)
        columns = {spec.name: pd.Series(dtype=_empty_dtype(spec.scalar)) for spec in specs}
        return cls.build(pd.DataFrame(columns))


def _empty_dtype(scalar: ColumnType) -> str:
    if scalar is int:
        return "int64"
    if scalar is float:
        return "float64"
    if scalar is bool:
        return "bool"
    return "object"


def _check_column(owner: str, series: pd.Series, spec: ColumnSpec) -> None:
    if spec.scalar is str:
        if not _is_string_column(series):
            example = _first_non_string(series)
            raise FrameSchemaError(
                f"{owner}.{spec.name} must be strings, found {series.dtype}"
                + (f" (example {example!r})" if example is not None else "")
            )
        if spec.allowed is not None:
            bad = ~series.isin(spec.allowed)
            if bool(bad.any()):
                example = series.loc[bad].iloc[0]
                choices = ", ".join(sorted(spec.allowed))
                raise FrameSchemaError(
                    f"{owner}.{spec.name} has value {example!r}, expected one of: {choices}"
                )
        return
    if spec.scalar is int and not pd.api.types.is_integer_dtype(series):
        raise FrameSchemaError(f"{owner}.{spec.name} must be an integer column, found {series.dtype}")
    if spec.scalar is float and not pd.api.types.is_float_dtype(series):
        raise FrameSchemaError(f"{owner}.{spec.name} must be a float column, found {series.dtype}")
    if spec.scalar is bool and not pd.api.types.is_bool_dtype(series):
        raise FrameSchemaError(f"{owner}.{spec.name} must be a boolean column, found {series.dtype}")


def _is_string_column(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.StringDtype):
        return bool(series.notna().all())
    if series.dtype != object:
        return False
    return all(isinstance(value, str) for value in series.to_numpy())


def _first_non_string(series: pd.Series) -> object | None:
    for value in series.to_numpy():
        if not isinstance(value, str):
            return value
    return None
