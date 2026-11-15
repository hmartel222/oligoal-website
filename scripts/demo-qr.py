#!/usr/bin/env python3
"""Print a QR code for the OligoAL site and save it as a PNG.

    python3 scripts/demo-qr.py https://oligoal.netlify.app

Point it at the stable Netlify URL, not the tunnel URL: the tunnel address
changes every restart, the Netlify one does not, so the QR stays valid for the
whole event. Needs `pip install segno`.
"""
import sys
from pathlib import Path

try:
    import segno
except ModuleNotFoundError:
    sys.exit("Install the QR library first:  pip install segno")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/demo-qr.py https://your-site.netlify.app")
    url = sys.argv[1].strip()
    if not url.startswith("https://"):
        sys.exit(f"Use the full https:// URL (got {url!r})")

    qr = segno.make(url, error="h")

    print()
    qr.terminal(compact=True)
    print(f"\n  {url}\n")

    out = Path("oligoal-qr.png")
    # scale 12 gives a ~500px image, large enough to project or print.
    qr.save(out, scale=12, border=4, light="#ffffff", dark="#0c2f34")
    print(f"  Saved {out.resolve()}")
    print("  Open it full screen or drop it on a slide for people to scan.\n")


if __name__ == "__main__":
    main()
