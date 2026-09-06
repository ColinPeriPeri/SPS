"""Supplier Problem Sheet (SPS) constrained-RAG pipeline."""

from .config import Settings
from .contracts import (
    SOLUTION_NOT_FOUND,
    Candidate,
    IncomingTicket,
    PipelineResult,
    SourceRecord,
)
from .pipeline import SPSPipeline

__version__ = "1.0.0"

__all__ = [
    "SOLUTION_NOT_FOUND",
    "Candidate",
    "IncomingTicket",
    "PipelineResult",
    "SPSPipeline",
    "Settings",
    "SourceRecord",
]
