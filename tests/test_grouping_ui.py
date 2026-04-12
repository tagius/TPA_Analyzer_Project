import asyncio
from pathlib import Path

import pandas as pd
from textual.widgets import Button, DataTable, Input, OptionList, Static

from tpa_analyzer.config.settings import AppSettings
from tpa_analyzer.core.constants import COMPUTED_METRICS
from tpa_analyzer.core.session import save_session_data, session_path
from tpa_analyzer.ui.app import TPAAnalyzerApp


def _make_app(tmp_path: Path) -> TPAAnalyzerApp:
    settings = AppSettings(default_data_dir=str(tmp_path), session_autosave_enabled=False)
    return TPAAnalyzerApp(settings=settings)


def test_loaded_files_start_unassigned(tmp_path: Path) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app._refresh_directory()
    assert [record["group"] for record in app.file_records] == [""]
    assert app.group_order == []


def test_sync_group_order_does_not_create_groups_from_file_assignments(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.file_records = [{"filename": "a.csv", "path": "a.csv", "group": "Alpha"}]
    app.group_order = []
    app._sync_group_order_from_records()
    assert app.group_order == []


def test_add_group_updates_group_order_without_duplicates(tmp_path: Path) -> None:
    app = _make_app(tmp_path)

    assert app._add_group(" Control ") is True
    assert app.group_order == ["Control"]

    assert app._add_group("Control") is False
    assert app.group_order == ["Control"]


def test_group_lifecycle_updates_group_order_and_assignments(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "a.csv", "path": "a.csv", "group": "Control"},
        {"filename": "b.csv", "path": "b.csv", "group": "Treatment"},
    ]
    app.group_order = ["Control", "Treatment"]

    assert app._rename_group("Treatment", "Heat") is True
    assert app.group_order == ["Control", "Heat"]
    assert [record["group"] for record in app.file_records] == ["Control", "Heat"]

    assert app._delete_group("Control") is True
    assert app.group_order == ["Heat"]
    assert [record["group"] for record in app.file_records] == ["", "Heat"]


def test_group_rename_invalidates_existing_analysis_results(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "a.csv").write_text("x", encoding="utf-8")
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            app.file_records = [
                {"filename": "a.csv", "path": str(tmp_path / "a.csv"), "group": "Control"}
            ]
            app.group_order = ["Control"]
            app.selected_group_order_index = 0
            app._render_file_list(selected_idx=0)
            app._render_group_order_list()
            app._apply_analysis_results(
                pd.DataFrame(
                    [
                        {
                            "Filename": "a.csv",
                            "Group": "Control",
                            **{metric: 1.0 for metric in COMPUTED_METRICS},
                        }
                    ]
                ),
                pd.DataFrame(
                    [
                        {
                            "Filename": "a.csv",
                            "Group": "Control",
                            "Time (s)": 0.0,
                            "Aligned Time (s)": 0.0,
                            "Force (N)": 1.0,
                            "Deformation (mm)": 0.1,
                        }
                    ]
                ),
                pd.DataFrame([{"Filename": "a.csv", "Group": "Control"}]),
                {
                    "Hardness (N)": {
                        "summary_df": pd.DataFrame(
                            [{"Group": "Control", "Mean": 1.0, "SD": 0.0, "Significance": "a"}]
                        ),
                        "pairwise_df": pd.DataFrame(),
                        "test_info": {"group_col": "Group", "group_order": ["Control"]},
                    }
                },
                [],
                [],
            )
            await pilot.pause()

            results_table = app.query_one("#results_table", DataTable)
            assert results_table.row_count == 1

            app.query_one("#input_group_name", Input).value = "Renamed"
            app.handle_group_rename()
            await pilot.pause()

            assert app.group_order == ["Renamed"]
            assert app.metrics_df.empty
            assert app.trace_df.empty
            assert app.qc_df.empty
            assert app.stats_results == {}
            assert results_table.row_count == 0
            assert "Run analysis again" in str(app.query_one("#status-msg", Static).render())

    asyncio.run(scenario())


def test_assign_selected_files_to_active_group(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "a.csv", "path": "a.csv", "group": ""},
        {"filename": "b.csv", "path": "b.csv", "group": ""},
        {"filename": "c.csv", "path": "c.csv", "group": "Other"},
    ]
    app.group_order = ["Control"]
    app.selected_group_order_index = 0
    app.selected_file_indices = {0, 2}

    app._assign_selected_files_to_active_group()
    assert [record["group"] for record in app.file_records] == ["Control", "", "Control"]


def test_clear_selected_file_assignments(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "a.csv", "path": "a.csv", "group": "Control"},
        {"filename": "b.csv", "path": "b.csv", "group": "Control"},
    ]
    app.selected_file_indices = {1}

    app._clear_selected_file_assignments()
    assert [record["group"] for record in app.file_records] == ["Control", ""]


def test_space_action_toggles_highlighted_file_selection(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
        (tmp_path / "sample_b.csv").write_text("x", encoding="utf-8")
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            file_table = app.query_one("#file_list", DataTable)
            file_table.focus()
            file_table.move_cursor(row=1, column=0, animate=False, scroll=True)
            await pilot.pause()

            await pilot.press("space")

            assert app.selected_file_index == 1
            assert app.selected_file_indices == {1}
            assert (
                str(app.query_one("#selected_file_count", Static).render()) == "Selected files: 1"
            )

    asyncio.run(scenario())


def test_invalid_refresh_clears_mounted_file_and_group_widgets(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            file_table = app.query_one("#file_list", DataTable)
            assert file_table.row_count == 1

            app.group_order = ["Control", "Treatment"]
            app.selected_group_order_index = 1
            app._render_group_order_list()
            await pilot.pause()

            group_list = app.query_one("#group_order_list", OptionList)
            assert group_list.option_count == 2
            assert app.query_one("#btn_group_up", Button).disabled is False

            app.query_one("#input_dir", Input).value = str(tmp_path / "missing")
            app._refresh_directory()
            await pilot.pause()

            assert file_table.row_count == 0
            assert str(app.query_one("#selected_file_info", Static).render()) == "Selected: none"
            assert group_list.option_count == 1
            assert app.query_one("#btn_group_up", Button).disabled is True
            assert app.query_one("#btn_group_down", Button).disabled is True

    asyncio.run(scenario())


def test_collect_session_payload_persists_active_group(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
        app = _make_app(tmp_path)

        async with app.run_test() as pilot:
            app.group_order = ["Control", "Treatment"]
            app.selected_group_order_index = 1
            app._render_group_order_list()
            await pilot.pause()

            payload = app._collect_session_payload()

            assert payload["active_group"] == "Treatment"

    asyncio.run(scenario())


def test_load_session_for_directory_ignores_missing_active_group(tmp_path: Path) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "sample_a.csv", "path": str(tmp_path / "sample_a.csv"), "group": ""}
    ]
    app.group_order = ["Legacy"]
    app.selected_group_order_index = 0

    save_session_data(
        session_path(tmp_path),
        {
            "group_order": ["Control"],
            "active_group": "Missing",
            "file_records": [{"filename": "sample_a.csv", "group": ""}],
        },
    )

    loaded, selected_idx = app._load_session_for_directory(tmp_path)

    assert loaded is True
    assert selected_idx is None
    assert app.group_order == ["Control"]
    assert app.selected_group_order_index is None


def test_load_session_for_directory_restores_existing_active_group(tmp_path: Path) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "sample_a.csv", "path": str(tmp_path / "sample_a.csv"), "group": ""}
    ]

    save_session_data(
        session_path(tmp_path),
        {
            "group_order": ["Control", "Treatment"],
            "active_group": "Treatment",
            "file_records": [{"filename": "sample_a.csv", "group": "Treatment"}],
        },
    )

    loaded, selected_idx = app._load_session_for_directory(tmp_path)

    assert loaded is True
    assert selected_idx is None
    assert app.group_order == ["Control", "Treatment"]
    assert app.file_records[0]["group"] == "Treatment"
    assert app.selected_group_order_index == 1


def test_load_session_for_directory_applies_saved_empty_group(tmp_path: Path) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "sample_a.csv", "path": str(tmp_path / "sample_a.csv"), "group": "Legacy"}
    ]

    save_session_data(
        session_path(tmp_path),
        {
            "group_order": ["Control"],
            "file_records": [{"filename": "sample_a.csv", "group": ""}],
        },
    )

    loaded, selected_idx = app._load_session_for_directory(tmp_path)

    assert loaded is True
    assert selected_idx is None
    assert app.group_order == ["Control"]
    assert app.file_records[0]["group"] == ""


def test_load_session_for_directory_derives_group_order_from_saved_records_when_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "sample_b.csv").write_text("x", encoding="utf-8")
    (tmp_path / "sample_c.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "sample_a.csv", "path": str(tmp_path / "sample_a.csv"), "group": ""},
        {"filename": "sample_b.csv", "path": str(tmp_path / "sample_b.csv"), "group": ""},
        {"filename": "sample_c.csv", "path": str(tmp_path / "sample_c.csv"), "group": ""},
    ]

    save_session_data(
        session_path(tmp_path),
        {
            "group_order": [],
            "file_records": [
                {"filename": "sample_a.csv", "group": "Beta"},
                {"filename": "sample_b.csv", "group": "Alpha"},
                {"filename": "sample_c.csv", "group": "Beta"},
            ],
        },
    )

    loaded, selected_idx = app._load_session_for_directory(tmp_path)

    assert loaded is True
    assert selected_idx is None
    assert app.group_order == ["Beta", "Alpha"]
    assert [record["group"] for record in app.file_records] == ["Beta", "Alpha", "Beta"]


def test_load_session_for_directory_clears_groups_missing_from_restored_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample_a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "sample_b.csv").write_text("x", encoding="utf-8")
    app = _make_app(tmp_path)
    app.file_records = [
        {"filename": "sample_a.csv", "path": str(tmp_path / "sample_a.csv"), "group": ""},
        {"filename": "sample_b.csv", "path": str(tmp_path / "sample_b.csv"), "group": ""},
    ]

    save_session_data(
        session_path(tmp_path),
        {
            "group_order": ["Control"],
            "active_group": "Treatment",
            "file_records": [
                {"filename": "sample_a.csv", "group": "Control"},
                {"filename": "sample_b.csv", "group": "Treatment"},
            ],
        },
    )

    loaded, selected_idx = app._load_session_for_directory(tmp_path)

    assert loaded is True
    assert selected_idx is None
    assert app.group_order == ["Control"]
    assert [record["group"] for record in app.file_records] == ["Control", ""]
    assert app.selected_group_order_index is None
