import asyncio
from pathlib import Path

import pandas as pd
from textual.widgets import Checkbox, Select

from tpa_analyzer.config.settings import AppSettings
from tpa_analyzer.core.models import GraphSpec
from tpa_analyzer.ui.app import TPAAnalyzerApp


def _make_app(tmp_path: Path) -> TPAAnalyzerApp:
    settings = AppSettings(default_data_dir=str(tmp_path), session_autosave_enabled=False)
    return TPAAnalyzerApp(settings=settings)


def _select_values(select: Select) -> list[str]:
    return [value for _, value in getattr(select, "_options", [])]


def _checkbox_by_label(app: TPAAnalyzerApp, label: str) -> Checkbox:
    for checkbox in app.query_one("#custom_left_axis_choices").query(Checkbox):
        if str(checkbox.label) == label:
            return checkbox
    raise AssertionError(f"Missing checkbox for {label!r}")


def test_overlay_control_disabled_before_analysis(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            overlay_select = app.query_one("#select_custom_overlay", Select)
            await pilot.pause()

            assert overlay_select.disabled is True
            assert _select_values(overlay_select) == ["__none__"]

    asyncio.run(scenario())


def test_right_axis_options_update_when_left_axis_selection_changes(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            app._apply_analysis_results(
                pd.DataFrame([{"Filename": "sample.csv", "Group": "Control"}]),
                pd.DataFrame(
                    [
                        {
                            "File": "sample.csv",
                            "Filename": "sample.csv",
                            "Group": "Control",
                            "Time (s)": 0.0,
                            "Aligned Time (s)": 0.0,
                            "Force (N)": 1.0,
                            "Force Corrected (N)": 1.0,
                            "Deformation (mm)": 0.1,
                        }
                    ]
                ),
                pd.DataFrame(),
                {},
                [],
                [],
            )
            await pilot.pause()

            right_axis = app.query_one("#select_custom_right_axis", Select)
            force = _checkbox_by_label(app, "Force (N)")
            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            deformation = _checkbox_by_label(app, "Deformation (mm)")

            force.value = False
            await pilot.pause()

            force_corrected.value = True
            await pilot.pause()
            assert _select_values(right_axis) == ["__none__", "Force (N)", "Deformation (mm)"]

            deformation.value = True
            await pilot.pause()
            assert _select_values(right_axis) == ["__none__", "Force (N)"]

            deformation.value = False
            await pilot.pause()
            assert _select_values(right_axis) == ["__none__", "Force (N)", "Deformation (mm)"]

    asyncio.run(scenario())


def test_legacy_graph_specs_render_recipe_style_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            app.graph_specs = [
                GraphSpec(
                    title="Legacy Trace",
                    plot_type="trace",
                    x_cols=["Time (s)"],
                    y_cols=["Force (N)", "Deformation (mm)"],
                )
            ]
            app._render_graph_specs()
            await pilot.pause()

            summary = str(app.query_one("#graph-spec-list").render())
            assert "Recipe: trace graph" in summary
            assert "Anchor domain: Time (s)" in summary
            assert "Primary axis: Force (N), Deformation (mm)" in summary
            assert "Secondary axis: None" in summary
            assert "Legacy trace recipe" not in summary
            assert "X domain:" not in summary
            assert "Left axis:" not in summary
            assert "Right axis:" not in summary

    asyncio.run(scenario())
