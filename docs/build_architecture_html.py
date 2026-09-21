"""Rebuild ARCHITECTURE.html by inlining the two diagrams into the explainer.

    python docs/build_architecture_html.py

ARCHITECTURE.html is self-contained so it can be mailed, opened offline or
printed to PDF, which means it holds a *copy* of each SVG. Edit a diagram and
the copy goes stale, so re-run this. It is stdlib-only and needs no network.

Two things have to be rewritten on the way in, and both are silent if missed:

  1. An SVG ``<style>`` block is not scoped once the SVG is inlined into HTML.
     Both diagrams define ``.h1`` / ``.h2`` / ``.bg`` with different values, so
     whichever landed second would restyle the first, and both would leak onto
     the page's own headings. Every rule is moved under the figure's id.
  2. ``<marker id="a">`` is likewise document-global, so ``url(#a)`` in one
     figure could resolve to the other's marker.
"""

import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent
TEMPLATE = DOCS / "architecture.template.html"
OUTPUT = DOCS / "ARCHITECTURE.html"

FIGURES = (
    ("<!--RUNTIME_SVG-->", "architecture-runtime.svg", "fig-runtime"),
    ("<!--TOOLING_SVG-->", "architecture-tooling.svg", "fig-tooling"),
)

_RULE = re.compile(r"^(\s*)\.([A-Za-z0-9_-]+)(\s*)\{", re.MULTILINE)


def inline(svg_path: pathlib.Path, fig_id: str) -> str:
    """Return the SVG with its CSS and marker ids scoped to `fig_id`."""
    svg = svg_path.read_text(encoding="utf-8").strip()

    svg = _RULE.sub(lambda m: f"{m.group(1)}#{fig_id} .{m.group(2)}{m.group(3)}{{", svg)

    for marker in re.findall(r'<marker\s+id="([^"]+)"', svg):
        svg = svg.replace(f'id="{marker}"', f'id="{fig_id}-{marker}"')
        svg = svg.replace(f"url(#{marker})", f"url(#{fig_id}-{marker})")

    # drop width/height so the page's CSS sizes it; viewBox keeps the ratio
    head = svg[: svg.index(">") + 1]
    new_head = re.sub(r'\s+(width|height)="[^"]*"', "", head)
    new_head = new_head.replace("<svg", f'<svg id="{fig_id}"', 1)
    return new_head + svg[len(head):]


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")

    for marker, name, fig_id in FIGURES:
        if marker not in html:
            print(f"{TEMPLATE.name}: missing {marker}", file=sys.stderr)
            return 1
        html = html.replace(marker, inline(DOCS / name, fig_id), 1)

    # nothing may have escaped scoping: a bare `.cls {` left in the body would
    # be a rule applying to the whole document
    body = html[html.index('<div class="sheet">'):]
    stray = sorted({m.group(2) for m in _RULE.finditer(body)})
    if stray:
        print("unscoped class rules leaked into the page: " + ", ".join(stray), file=sys.stderr)
        return 1

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(DOCS.parent)} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
