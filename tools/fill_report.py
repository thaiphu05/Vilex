"""Fill [[KEY]] placeholders of the Vilex report .docx from report_numbers.json.

    python3 tools/fill_report.py --docx reports/bao-cao-vilex-tong-ket.docx \
        --numbers stats_out/report_numbers.json --out reports/bao-cao-vilex-tong-ket-final.docx

Only the text and the yellow highlight of runs holding a known placeholder are
changed. Unknown keys stay highlighted and are listed; exit code 1 when any
remain (``--blank-missing`` replaces them with "-" instead). The input .docx is
never modified and the output path must not exist. Standard library only.
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

RUN = re.compile(r"<w:r(?:\s[^>]*)?>.*?</w:r>", re.S)
KEY = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]")
HIGHLIGHT = re.compile(r"<w:highlight\b[^>]*/>")


def fill_xml(xml, numbers, blank_missing=False):
    """Return (new_xml, filled_keys, missing_keys)."""
    filled, missing = set(), set()

    def run(match):
        text = match.group(0)
        if "[[" not in text:
            return text
        still_missing = False

        def sub(m):
            nonlocal still_missing
            k = m.group(1)
            if k in numbers:
                filled.add(k)
                return escape(str(numbers[k]))
            missing.add(k)
            if blank_missing:
                return "-"
            still_missing = True
            return m.group(0)

        text = KEY.sub(sub, text)
        return text if still_missing else HIGHLIGHT.sub("", text)

    return RUN.sub(run, xml), filled, missing


def fill_docx(src, dst, numbers, blank_missing=False):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        filled, missing = set(), set()
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml, filled, missing = fill_xml(data.decode("utf-8"), numbers, blank_missing)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    return filled, missing


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--docx", required=True, type=Path)
    ap.add_argument("--numbers", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blank-missing", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"Output already exists: {args.out}")
        numbers = json.loads(args.numbers.read_text(encoding="utf-8"))
        filled, missing = fill_docx(args.docx, args.out, numbers, args.blank_missing)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Filled {len(filled)} keys -> {args.out}")
    if missing:
        print("Missing keys: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 0 if args.blank_missing else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
