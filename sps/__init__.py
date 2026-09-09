"""Supplier Problem Sheet (SPS) constrained-RAG pipeline."""

from .contracts import SOLUTION_NOT_FOUND, Candidate, IncomingTicket, PipelineResult
from .validators import normalize_part_number, validate_ticket

__version__ = "2.0.0"

__all__ = [
    "SOLUTION_NOT_FOUND",
    "Candidate",
    "IncomingTicket",
    "PipelineResult",
    "normalize_part_number",
    "validate_ticket",
]
