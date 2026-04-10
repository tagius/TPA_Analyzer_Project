import asyncio
from pathlib import Path

import pandas as pd
from textual.widgets import Checkbox, OptionList, Select

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


def _builder_option_prompts(option_list: OptionList) -> list[str]:
    return [str(option.prompt) for option in option_list._options]


def test_segment_control_disabled_before_analysis(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            segment_select = app.query_one("#select_custom_segment", Select)
            await pilot.pause()

            assert segment_select.disabled is True
            assert _select_values(segment_select) == ["__none__"]

    asyncio.run(scenario())


def test_segment_options_stay_disabled_without_qc_markers(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is True
            assert _select_values(segment_select) == ["__none__"]

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
                pd.DataFrame([{"Filename": "sample.csv", "Group": "Control"}]),
                {},
                [],
                [],
            )
            await pilot.pause()

            app.query_one("#select_custom_view_domain", Select).value = "semantic_segment"
            await pilot.pause()

            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            force_corrected.value = True
            await pilot.pause()

            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is True
            assert _select_values(segment_select) == ["__none__"]

    asyncio.run(scenario())


def test_segment_options_enable_when_qc_markers_are_available(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is True
            assert _select_values(segment_select) == ["__none__"]

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
                pd.DataFrame(
                    [
                        {
                            "Filename": "sample.csv",
                            "Group": "Control",
                            "Bite1 Start Index": 0,
                            "Peak1 Index": 0,
                        }
                    ]
                ),
                {},
                [],
                [],
            )
            await pilot.pause()

            app.query_one("#select_custom_view_domain", Select).value = "semantic_segment"
            await pilot.pause()

            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            force_corrected.value = True
            await pilot.pause()

            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is False
            assert "b1_start_to_peak1" in _select_values(segment_select)

    asyncio.run(scenario())


def test_segment_selection_triggers_one_autosave_when_builder_is_synchronized(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = TPAAnalyzerApp(settings=AppSettings(default_data_dir=str(tmp_path), session_autosave_enabled=True))
        autosave_calls = 0
        original_autosave = app._autosave_session

        def counting_autosave() -> None:
            nonlocal autosave_calls
            autosave_calls += 1
            original_autosave()

        app._autosave_session = counting_autosave  # type: ignore[method-assign]

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
                pd.DataFrame(
                    [
                        {
                            "Filename": "sample.csv",
                            "Group": "Control",
                            "Bite1 Start Index": 0,
                            "Peak1 Index": 0,
                        }
                    ]
                ),
                {},
                [],
                [],
            )
            await pilot.pause()

            app.query_one("#select_custom_view_domain", Select).value = "semantic_segment"
            await pilot.pause()

            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            force_corrected.value = True
            await pilot.pause()

            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is False
            assert "b1_start_to_peak1" in _select_values(segment_select)

            autosave_calls = 0
            segment_select.value = "b1_start_to_peak1"
            await pilot.pause()

            assert autosave_calls == 1

    asyncio.run(scenario())


def test_segment_options_reset_when_grouping_change_invalidates_analysis_results(tmp_path: Path) -> None:
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
                pd.DataFrame(
                    [
                        {
                            "Filename": "sample.csv",
                            "Group": "Control",
                            "Bite1 Start Index": 0,
                            "Peak1 Index": 0,
                        }
                    ]
                ),
                {},
                [],
                [],
            )
            await pilot.pause()

            app.query_one("#select_custom_view_domain", Select).value = "semantic_segment"
            await pilot.pause()

            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            force_corrected.value = True
            await pilot.pause()

            segment_select = app.query_one("#select_custom_segment", Select)
            assert segment_select.disabled is False
            assert "b1_start_to_peak1" in _select_values(segment_select)

            invalidated = app._invalidate_analysis_results_for_grouping_change()
            await pilot.pause()

            segment_select = app.query_one("#select_custom_segment", Select)
            assert invalidated is True
            assert segment_select.disabled is True
            assert _select_values(segment_select) == ["__none__"]

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
            assert deformation.display is False

            deformation.value = True
            await pilot.pause()
            assert deformation.value is False
            assert _select_values(right_axis) == ["__none__", "Force (N)", "Deformation (mm)"]

            force_corrected.value = False
            await pilot.pause()
            assert _select_values(right_axis) == ["__none__", "Force (N)", "Force Corrected (N)", "Deformation (mm)"]
            assert deformation.display is True

    asyncio.run(scenario())


def test_left_axis_incompatible_units_are_hidden_when_force_is_selected(tmp_path: Path) -> None:
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

            force = _checkbox_by_label(app, "Force (N)")
            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            deformation = _checkbox_by_label(app, "Deformation (mm)")

            assert force.value is True
            assert force_corrected.display is True
            assert deformation.display is False

            force.value = False
            await pilot.pause()

            assert force_corrected.display is True
            assert deformation.display is True

    asyncio.run(scenario())


def test_segment_domain_enables_segment_control_and_filters_annotations(tmp_path: Path) -> None:
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
                pd.DataFrame(
                    [
                        {
                            "Filename": "sample.csv",
                            "Group": "Control",
                            "Bite1 Start Index": 0,
                            "Peak1 Index": 1,
                            "Bite1 End Index": 2,
                            "Bite2 Start Index": 3,
                        }
                    ]
                ),
                {},
                [],
                [],
            )
            await pilot.pause()

            force = _checkbox_by_label(app, "Force (N)")
            force.value = False
            await pilot.pause()

            force_corrected = _checkbox_by_label(app, "Force Corrected (N)")
            force_corrected.value = True
            await pilot.pause()

            view_domain = app.query_one("#select_custom_view_domain", Select)
            segment = app.query_one("#select_custom_segment", Select)
            annotations = app.query_one("#select_custom_annotation", Select)

            assert str(view_domain.value) == "full_curve"
            assert segment.disabled is True
            assert _select_values(annotations) == ["__none__"]

            view_domain.value = "semantic_segment"
            await pilot.pause()

            assert segment.disabled is False
            assert "b1_start_to_peak1" in _select_values(segment)
            segment.value = "b1_start_to_peak1"
            await pilot.pause()
            assert _select_values(annotations) == ["__none__", "hardness_peak1"]

            segment.value = "b1_end_to_b2_start"
            await pilot.pause()

            assert _select_values(annotations) == ["__none__", "adhesiveness"]

    asyncio.run(scenario())


def test_selected_samples_scope_uses_dedicated_plot_builder_list(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            app._apply_analysis_results(
                pd.DataFrame(
                    [
                        {"Filename": "sample-a.csv", "Group": "Control"},
                        {"Filename": "sample-b.csv", "Group": "Treatment"},
                    ]
                ),
                pd.DataFrame(
                    [
                        {
                            "File": "sample-a.csv",
                            "Filename": "sample-a.csv",
                            "Group": "Control",
                            "Time (s)": 0.0,
                            "Aligned Time (s)": 0.0,
                            "Force (N)": 1.0,
                            "Force Corrected (N)": 1.0,
                            "Deformation (mm)": 0.1,
                        },
                        {
                            "File": "sample-b.csv",
                            "Filename": "sample-b.csv",
                            "Group": "Treatment",
                            "Time (s)": 0.0,
                            "Aligned Time (s)": 0.0,
                            "Force (N)": 1.2,
                            "Force Corrected (N)": 1.2,
                            "Deformation (mm)": 0.2,
                        },
                    ]
                ),
                pd.DataFrame(
                    [
                        {"Filename": "sample-a.csv", "Group": "Control", "Bite1 Start Index": 0, "Peak1 Index": 1},
                        {"Filename": "sample-b.csv", "Group": "Treatment", "Bite1 Start Index": 0, "Peak1 Index": 1},
                    ]
                ),
                {},
                [],
                [],
            )
            await pilot.pause()

            data_scope = app.query_one("#select_custom_data_scope", Select)
            sample_list = app.query_one("#custom_graph_sample_list", OptionList)

            assert data_scope.value == "grouped"
            assert sample_list.disabled is True
            assert _builder_option_prompts(sample_list) == ["sample-a.csv", "sample-b.csv"]

            data_scope.value = "selected_samples"
            await pilot.pause()

            assert sample_list.disabled is False

            sample_list.highlighted = 0
            sample_list.action_select()
            await pilot.pause()

            spec = app._collect_graph_spec_from_ui()
            assert spec.data_scope == "selected_samples"
            assert spec.selected_samples == ["sample-a.csv"]

    asyncio.run(scenario())


def test_export_all_uses_filtered_metrics_payload(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        captured: dict[str, pd.DataFrame | list[str]] = {}

        async with app.run_test() as pilot:
            app.trace_df = pd.DataFrame(
                [
                    {"Filename": "blank.csv", "File": "blank.csv", "Group": "", "Time (s)": 0.0, "Force (N)": 1.0},
                    {"Filename": "control.csv", "File": "control.csv", "Group": "Control", "Time (s)": 0.0, "Force (N)": 2.0},
                ]
            )
            app.metrics_df = pd.DataFrame(
                [
                    {"Filename": "blank.csv", "Group": "", "Hardness (N)": 1.0},
                    {"Filename": "control.csv", "Group": "Control", "Hardness (N)": 2.0},
                ]
            )
            app.qc_df = pd.DataFrame(
                [
                    {"Filename": "blank.csv", "Group": ""},
                    {"Filename": "control.csv", "Group": "Control"},
                ]
            )
            app.stats_results = {}
            app.group_order = ["", "Control"]
            app.graph_specs = [
                GraphSpec(
                    title="Metric Graph",
                    plot_type="metric",
                    x_cols=["Group"],
                    y_cols=["Hardness (N)"],
                )
            ]

            def fake_export_all_worker(**kwargs) -> None:
                captured.update(kwargs)

            monkeypatch.setattr(app, "export_all_worker", fake_export_all_worker)

            app.trigger_export_all()
            await pilot.pause()

            assert list(captured["metrics_df"]["Group"]) == ["Control"]
            assert captured["group_order"] == ["Control"]

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
