"""The install stays small, and these keep it that way.

Every package is one more thing that can fail to install, or be blocked, on a
managed Windows machine: Application Control has already stopped two native
DLLs here, jiter 0.17's and lxml 6's. So the code may import the standard
library and four packages, and requirements.txt may list those four and the two
pins they need, plus the test runner.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = {"numpy", "openai", "openpyxl", "pydantic"}
PINNED_FOR_OPENAI = {"httpx", "jiter"}
TEST_RUNNER = {"pytest", "pytest-asyncio"}
PROJECT = {"sps", "sps_prompts"}


def _imported_top_level(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_code_imports_only_the_standard_library_and_four_packages():
    for name in ("sps.py", "sps_prompts.py"):
        outside = _imported_top_level(ROOT / name) - set(sys.stdlib_module_names) - PROJECT
        assert outside <= RUNTIME, f"{name} imports {sorted(outside - RUNTIME)}"


def test_requirements_list_nothing_more():
    listed = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if requirement:
            listed.add(re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].lower())
    assert listed == RUNTIME | PINNED_FOR_OPENAI | TEST_RUNNER
