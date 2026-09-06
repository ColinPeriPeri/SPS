"""Component A.1 -- delta reader over the source relational database.

Keyset pagination on (Last_Modified_Date, SPS_ID) rather than OFFSET: constant
cost at any depth, and stable even if rows are written during a long backfill.
"""

from __future__ import annotations

import re
from typing import Iterator, Protocol, Sequence, runtime_checkable

from ..contracts import SourceRecord
from .watermark import Watermark

# Column mapping: source column -> contract field.
COLUMN_MAP: dict[str, str] = {
    "SPS_ID": "sps_id",
    "Problem_Description": "problem_description",
    "Actual_Solution": "actual_solution",
    "Part_Number": "part_number",
    "Part_Description": "part_description",
    "Item_Status": "item_status",
    "Problem_Reason_Code": "problem_reason_code",
    "Issue_Type": "issue_type",
    "Last_Modified_Date": "last_modified_date",
}

_SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


@runtime_checkable
class RecordSource(Protocol):
    def fetch_since(self, watermark: Watermark, chunk_size: int) -> Iterator[SourceRecord]:
        """Yield records with (Last_Modified_Date, SPS_ID) > watermark, ascending."""


class SqlRecordSource:
    """SQLAlchemy delta reader."""

    def __init__(self, dsn: str, table: str, engine=None) -> None:
        if not _SAFE_TABLE.match(table):
            # The table name is interpolated (identifiers cannot be bound), so
            # it is validated against a strict allowlist pattern first.
            raise ValueError(f"Unsafe table identifier: {table!r}")
        self.dsn = dsn
        self.table = table
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(self.dsn, pool_pre_ping=True)
        return self._engine

    def _query(self, chunk_size: int):
        from sqlalchemy import text

        columns = ", ".join(COLUMN_MAP)
        return text(
            f"SELECT TOP (:chunk) {columns} FROM {self.table} "
            "WHERE Last_Modified_Date > :wm_date "
            "   OR (Last_Modified_Date = :wm_date AND SPS_ID > :wm_id) "
            "ORDER BY Last_Modified_Date ASC, SPS_ID ASC"
        )

    def fetch_since(self, watermark: Watermark, chunk_size: int) -> Iterator[SourceRecord]:
        cursor = watermark
        statement = self._query(chunk_size)
        while True:
            with self.engine.connect() as connection:
                rows = connection.execute(
                    statement,
                    {
                        "chunk": chunk_size,
                        "wm_date": cursor.last_modified_date,
                        "wm_id": cursor.sps_id,
                    },
                ).mappings().all()
            if not rows:
                return
            for row in rows:
                record = SourceRecord.from_row(
                    {field: row[column] for column, field in COLUMN_MAP.items()}
                )
                yield record
            last = rows[-1]
            cursor = Watermark(
                last_modified_date=last["Last_Modified_Date"],
                sps_id=str(last["SPS_ID"]),
            )
            if len(rows) < chunk_size:
                return


class InMemoryRecordSource:
    """Test/dev double honouring the same keyset contract."""

    def __init__(self, records: Sequence[SourceRecord]) -> None:
        self.records = sorted(records, key=lambda r: r.sort_key())

    def fetch_since(self, watermark: Watermark, chunk_size: int) -> Iterator[SourceRecord]:
        cutoff = (watermark.last_modified_date, watermark.sps_id)
        for record in self.records:
            if record.sort_key() > cutoff:
                yield record
