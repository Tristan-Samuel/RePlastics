#!/usr/bin/env python3
"""Refuse the old Drive-checkpoint fallback.

Training uploads the checkpoint you named from this computer.
This script used to copy a different Drive file when the named one
was missing. That substituted the September head for the 5 Oct model.
It now stops instead of choosing another file.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Refusing to load a checkpoint from Google Drive. "
        "colab_train.sh uploads the --model file from this computer "
        "and does not substitute another one.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
