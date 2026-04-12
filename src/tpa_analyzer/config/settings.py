"""Environment-backed application settings."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean-like environment variable safely."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class AppSettings:
    """Resolved runtime and developer defaults for the application."""

    log_level: str = "INFO"
    default_data_dir: Path = Path(".")
    export_root_name: str = "exports"
    plots_root_name: str = "output_plots"
    session_autosave_enabled: bool = True
    debug_enabled: bool = False

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Build application settings from environment variables."""
        default_dir = Path(os.getenv("TPA_ANALYZER_DEFAULT_DATA_DIR", ".")).expanduser()
        return cls(
            log_level=os.getenv("TPA_ANALYZER_LOG_LEVEL", "INFO").upper(),
            default_data_dir=default_dir,
            export_root_name=os.getenv("TPA_ANALYZER_EXPORT_ROOT", "exports"),
            plots_root_name=os.getenv("TPA_ANALYZER_PLOTS_ROOT", "output_plots"),
            session_autosave_enabled=_parse_bool(
                os.getenv("TPA_ANALYZER_SESSION_AUTOSAVE"),
                True,
            ),
            debug_enabled=_parse_bool(os.getenv("TPA_ANALYZER_DEBUG"), False),
        )
