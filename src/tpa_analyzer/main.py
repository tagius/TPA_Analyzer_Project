"""Application entrypoint helpers."""

from __future__ import annotations

from tpa_analyzer.config.settings import AppSettings
from tpa_analyzer.ui.app import TPAAnalyzerApp


def run() -> None:
    """Run the TPA Analyzer Textual application."""
    app = TPAAnalyzerApp(settings=AppSettings.from_env())
    app.run()
