"""Render the short reports these scripts write. Headings, lists, and tables."""

import html
import re

from markupsafe import Markup


def _inline(text):
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _table(rows):
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        parsed.append(cells)
    if len(parsed) < 2:
        return "<p>" + "<br>".join(_inline(row) for row in rows) + "</p>"
    header, body = parsed[0], parsed[2:]
    parts = ["<table><thead><tr>"]
    for cell in header:
        parts.append(f"<th>{_inline(cell)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{_inline(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _is_separator(line):
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line))


def to_html(text):
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("#"):
            marks = len(line) - len(line.lstrip("#"))
            level = min(max(marks, 1), 4)
            blocks.append(
                f"<h{level}>{_inline(line[marks:].strip())}</h{level}>"
            )
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and _is_separator(lines[index + 1])
        ):
            rows = []
            while index < len(lines) and "|" in lines[index]:
                rows.append(lines[index])
                index += 1
            blocks.append(_table(rows))
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(f"<li>{_inline(lines[index][2:])}</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        chunk = []
        while index < len(lines) and lines[index].strip():
            if lines[index].startswith("#") or lines[index].startswith("- "):
                break
            if (
                "|" in lines[index]
                and index + 1 < len(lines)
                and _is_separator(lines[index + 1])
            ):
                break
            chunk.append(lines[index])
            index += 1
        blocks.append("<p>" + "<br>".join(_inline(bit) for bit in chunk) + "</p>")
    return Markup("\n".join(blocks))


def section(text, heading):
    """Return the body under one heading, up to the next heading of the same rank or higher."""
    lines = text.splitlines()
    start = None
    level = None
    for index, line in enumerate(lines):
        if line.startswith("#") and line.lstrip("#").strip() == heading:
            level = len(line) - len(line.lstrip("#"))
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        line = lines[index]
        if not line.startswith("#"):
            continue
        marks = len(line) - len(line.lstrip("#"))
        if marks <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()
