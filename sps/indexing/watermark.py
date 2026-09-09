"""LEGACY PATH -- not on the resolver flow.

The primary entry point is now `scripts/run_resolver.py`, which filters a
history file by part number and embeds the survivors per ticket, so there is
no persistent index to build or maintain. This module is retained for
`service/run_inference.py` and for a future return to a persistent index;
nothing on the resolver path imports it.

Component A.1 -- high-water mark persistence.

The mark is a composite (Last_Modified_Date, SPS_ID). A plain timestamp would
skip records that share the boundary second; the tuple makes resumption exact.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Watermark:
    last_modified_date: datetime
    sps_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "last_modified_date": self.last_modified_date.isoformat(),
            "sps_id": self.sps_id,
        }

    @classmethod
    def initial(cls) -> "Watermark":
        return cls(last_modified_date=EPOCH, sps_id="")


class WatermarkStore:
    """Atomic file-backed watermark.

    Written via temp-file + os.replace so a crash mid-write cannot leave a
    truncated mark that would silently re-index (or skip) the whole table.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> Watermark:
        if not self.path.exists():
            return Watermark.initial()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Watermark(
                last_modified_date=datetime.fromisoformat(data["last_modified_date"]),
                sps_id=str(data.get("sps_id", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupt mark: fall back to a full rebuild rather than guess.
            return Watermark.initial()

    def write(self, watermark: Watermark) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(watermark.as_dict(), stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
