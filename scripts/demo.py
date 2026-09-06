"""Offline end-to-end demonstration.

Runs Components A, B and C against an in-memory vector store, a deterministic
stand-in embedder and a scripted LLM, so the whole flow -- including every
failure path -- can be observed with no Qdrant, no Azure and no model download.

    python -m scripts.demo
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from sps.config import IndexingSettings, LLMSettings, RetrievalSettings, Settings
from sps.contracts import IncomingTicket, SourceRecord
from sps.generation import ActorCriticLoop
from sps.indexing import IncrementalIndexer, InMemoryRecordSource
from sps.pipeline import SPSPipeline
from sps.retrieval import Retriever
from sps.vectorstore import InMemoryVectorStore

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tests.conftest import ScriptedChatClient, TokenOverlapEmbedder  # noqa: E402

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def historical_records() -> list[SourceRecord]:
    def record(sps_id, problem, solution, minutes, **meta):
        fields = {
            "part_number": "PN-1000",
            "part_description": "Mounting bracket",
            "item_status": "Active",
            "problem_reason_code": "RC-WELD",
            "issue_type": "Quality",
        }
        fields.update(meta)
        return SourceRecord(
            sps_id=sps_id,
            problem_description=problem,
            actual_solution=solution,
            last_modified_date=BASE + timedelta(minutes=minutes),
            **fields,
        )

    return [
        record(
            "SPS-1001",
            "Bracket weld seam cracking observed during incoming inspection",
            "Rework the weld seam per the original joint profile. Re-inspect the seam "
            "before shipment and include the inspection record with the delivery.",
            0,
        ),
        record(
            "SPS-1002",
            "Weld seam cracks found on mounting bracket at receiving",
            "Segregate the affected lot. Rework the cracked seams and re-inspect. "
            "Ship replacements for any bracket that fails re-inspection.",
            10,
        ),
        record(
            "SPS-1003",
            "Outer carton label misprint on shipment packaging",
            "Reprint the carton labels and re-apply before dispatch.",
            20,
            part_number="PN-2000",
            problem_reason_code="RC-LABEL",
            issue_type="Packaging",
        ),
        # Dropped by sanitization: solution under 15 characters.
        record("SPS-1004", "Surface corrosion noted on the flange face", "TBD", 30),
        # Duplicate of SPS-1001 with a later timestamp -- SPS-1001 is superseded.
        record(
            "SPS-1099",
            "Bracket weld seam cracking observed during incoming inspection",
            "Rework the weld seam per the original joint profile. Re-inspect the seam "
            "before shipment and include the inspection record with the delivery.",
            40,
        ),
    ]


def build_index(embedder, store, tmp_watermark: str):
    report = IncrementalIndexer(
        source=InMemoryRecordSource(historical_records()),
        embedder=embedder,
        store=store,
        settings=IndexingSettings(batch_size=250, watermark_path=tmp_watermark),
    ).run()
    return report


def pipeline_with(embedder, store, responses):
    settings = Settings(retrieval=RetrievalSettings(), llm=LLMSettings())
    return SPSPipeline(
        retriever=Retriever(embedder, store, settings.retrieval),
        loop=ActorCriticLoop(ScriptedChatClient(responses), settings.llm),
        settings=settings,
    )


def actor(recommendation, justification):
    return json.dumps({"recommendation": recommendation, "justification": justification})


PASS = json.dumps({"status": "PASS"})
FAIL_HALLUCINATION = json.dumps(
    {
        "status": "FAIL",
        "critique": "'Preheat to 150 C' does not appear in the historical solutions. "
        "Remove the preheat step rather than replacing it.",
    }
)
FAIL_LEAKAGE = json.dumps(
    {
        "status": "FAIL",
        "critique": "'Log the disposition in the internal MES quality module' directs "
        "an external supplier to an internal system. Remove that step.",
    }
)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    import tempfile, os

    tmp = os.path.join(tempfile.mkdtemp(), "watermark.json")

    embedder = TokenOverlapEmbedder()
    store = InMemoryVectorStore()

    print("=" * 78)
    print("COMPONENT A -- incremental indexing")
    print("=" * 78)
    report = build_index(embedder, store, tmp)
    print(json.dumps(report.as_dict(), indent=2))
    print(f"\nindexed points in store: {store.count()}")
    print("(SPS-1004 dropped: solution under 15 chars. "
          "SPS-1001 superseded by the later duplicate SPS-1099.)\n")

    print("A second run with no source changes indexes nothing:")
    second = build_index(embedder, store, tmp)
    print(f"  batches={second.batches} indexed={second.indexed}\n")

    print("A later run submitting the SAME problem+solution under a NEW SPS_ID is")
    print("caught by the persisted content_hash (cross-run dedup):")
    duplicate = SourceRecord(
        sps_id="SPS-2001",
        problem_description="Bracket weld seam cracking observed during incoming inspection",
        actual_solution="Rework the weld seam per the original joint profile. Re-inspect "
        "the seam before shipment and include the inspection record with the delivery.",
        part_number="PN-1000",
        part_description="Mounting bracket",
        item_status="Active",
        problem_reason_code="RC-WELD",
        issue_type="Quality",
        last_modified_date=BASE + timedelta(days=7),
    )
    later = IncrementalIndexer(
        source=InMemoryRecordSource(historical_records() + [duplicate]),
        embedder=embedder,
        store=store,
        settings=IndexingSettings(batch_size=250, watermark_path=tmp),
    ).run()
    print(
        f"  indexed={later.indexed} "
        f"cross_run_duplicates={later.cross_run_duplicates} "
        f"-> points still {store.count()} (SPS-1099 replaced by SPS-2001)\n"
    )

    scenarios = [
        (
            "1. Strong match, passes the audit first try",
            {"problem_description": "Bracket weld seam cracking observed during incoming inspection",
             "part_number": "PN-1000", "issue_type": "Quality", "problem_reason_code": "RC-WELD"},
            [actor("1. Segregate the affected lot.\n"
                   "2. Rework the cracked weld seam per the original joint profile.\n"
                   "3. Re-inspect the seam before shipment and include the inspection record.",
                   "SPS-2001 records the same weld seam cracking on this bracket and was "
                   "resolved by rework and re-inspection before shipment."),
             PASS],
        ),
        (
            "2. Hallucinated step caught, then refined and passed",
            {"problem_description": "Weld seam cracks found on mounting bracket at receiving",
             "part_number": "PN-1000", "issue_type": "Quality"},
            [actor("1. Preheat to 150 C.\n2. Rework the weld seam.", "Drawn from SPS-1002."),
             FAIL_HALLUCINATION,
             actor("1. Segregate the affected lot.\n2. Rework the cracked seams and re-inspect.",
                   "Drawn from SPS-1002, the same weld seam defect on this bracket."),
             PASS],
        ),
        (
            "3. Internal tool leakage caught, then refined and passed",
            {"problem_description": "Weld seam cracks found on mounting bracket at receiving",
             "part_number": "PN-1000"},
            [actor("1. Rework the seams.\n2. Log the disposition in the internal MES quality module.",
                   "Drawn from SPS-1002."),
             FAIL_LEAKAGE,
             actor("1. Segregate the affected lot.\n2. Rework the cracked seams and re-inspect.",
                   "Drawn from SPS-1002."),
             PASS],
        ),
        (
            "4. Confidence gate blocks a weak match (no LLM call)",
            {"problem_description": "Hydraulic pump pressure fluctuating during the acceptance run",
             "part_number": "PN-7777"},
            [],
        ),
        (
            "5. Invalid input rejected before retrieval",
            {"problem_description": "cracked"},
            [],
        ),
        (
            "6. Circuit breaker trips after 3 failed audits",
            {"problem_description": "Bracket weld seam cracking observed during incoming inspection",
             "part_number": "PN-1000"},
            [actor("1. Preheat to 150 C.", "x"), FAIL_HALLUCINATION,
             actor("1. Preheat to 200 C.", "x"), FAIL_HALLUCINATION,
             actor("1. Preheat to 250 C.", "x"), FAIL_HALLUCINATION],
        ),
    ]

    print("=" * 78)
    print("COMPONENTS B + C -- query, gate, actor-critic loop")
    print("=" * 78)
    for title, ticket, responses in scenarios:
        pipeline = pipeline_with(embedder, store, responses)
        client = pipeline.loop.client
        result = await pipeline.process_dict(ticket)
        print(f"\n--- {title}")
        print(f"    problem: {ticket['problem_description'][:70]}")
        print(f"    LLM calls made: {client.call_count}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
