"""Launcher for the pronunciation-coach user-trial UI.

Loads models fail-fast, then serves locally only.
"""

import sys

from ui.interface import build_app
from ui.models import StartupError, load_models


def main() -> None:
    try:
        models = load_models()
    except StartupError as e:
        print(f"\nStartup failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"app_session_id: {models.app_session_id}")
    print(f"trial logs: {models.trial_logs_dir}")
    demo = build_app(models)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
