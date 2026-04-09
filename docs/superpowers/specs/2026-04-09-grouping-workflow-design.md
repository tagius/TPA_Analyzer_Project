# Grouping Workflow Redesign

## Summary

Redesign the Textual grouping workflow so users manage an explicit list of groups first, then assign one or many selected files to an active group. The new flow removes filename-based inference and batch term matching, keeps manual group ordering, and reuses the existing downstream analysis, plotting, export, and color logic where possible.

## Goals

- Replace free-form per-file grouping with an explicit managed group list.
- Support assigning multiple files to one group in a single action.
- Keep group ordering as a manual user-controlled list used by plots and statistics.
- Start every loaded file as unassigned.
- Remove unused filename inference and batch matching logic if they are no longer needed.

## Non-Goals

- No automatic group creation from filenames.
- No filename-term batch assignment workflow.
- No broader redesign of analysis, plotting, statistics, or export behavior beyond what is required to preserve compatibility with the new grouping UI.

## User Workflow

1. User loads a directory of supported files.
2. Every file starts in an unassigned state.
3. User creates one or more groups in a dedicated group list.
4. User selects one active group from the group list.
5. User selects one or many files in the file table.
6. User runs an assignment action to attach the selected files to the active group.
7. User may clear assignments, rename groups, delete groups, or reorder groups.
8. Deleting a group moves its assigned files back to `UNASSIGNED`.

`UNASSIGNED` is a system display state for files without a group. It is not a normal user-managed group in the editable group list.

## Recommended Approach

Use an explicit group-management panel plus a multi-select file table.

This approach fits the current structure in `src/tpa_analyzer/ui/app.py`, keeps the current left-pane grouping area concept, and preserves the existing downstream contract: each file still resolves to a group string, and `group_order` remains the source of truth for ordering in statistics, plots, and color selection.

Other options considered but rejected:

- File-centric inline editing in the file table: efficient for row edits, but less clear for rename/delete/reorder semantics in a terminal UI.
- Dual-list transfer layout: visually explicit, but a worse fit for the current `DataTable`-based file overview and existing layout.

## UI Structure

Retain the left pane, but split its responsibilities into two clearer sections.

### Group Management Section

- A `Group List` widget shows only real user-managed groups.
- Actions:
  - `Add`
  - `Rename`
  - `Delete`
  - `Up`
  - `Down`
- One highlighted or selected group is the active target for file assignment.
- The same managed group set remains the source for color customization.

### File Assignment Section

- Keep the detected files table as the main file overview.
- Show filename and current group assignment, with unassigned files rendered as `UNASSIGNED`.
- Add multi-selection support so one action can assign many files to the active group.
- Replace the current grouping actions with:
  - `Assign to Active Group`
  - `Clear Assignment`
- Remove:
  - free-form `Group Name` input
  - `Batch Match Terms`
  - `Assign Terms`

### Selection Behavior

Primary target behavior:

- Mouse-driven multi-selection using standard modifier semantics such as `Shift` and `Ctrl`, if the chosen Textual widget supports this cleanly.
- Keyboard fallback as a secondary path when modifier handling is limited by the terminal or widget behavior.

Required supporting UI cues:

- a visible selected-file count
- clear indication of the active target group
- disabled assignment actions when no target group exists or no files are selected

## Data Model

Keep `file_records` as the main per-file source of truth, but stop using it to invent groups dynamically.

### File State

- `file_records` continues to hold file-level data such as filename, path, and assigned group string.
- On refresh, each file record starts with an empty internal group value and is displayed as `UNASSIGNED`.

### Group State

- `group_order` continues to represent the ordered list of real groups.
- `group_order` is now modified only through explicit group-management actions:
  - add
  - rename
  - delete
  - reorder
- File assignment does not create groups implicitly.

### UI State

Add separate UI state for:

- active group index
- selected file indices

Persist enough state in the session payload to restore:

- managed groups
- per-file assignments
- active group if that group still exists after session restore

Persisting the current multi-selection set is not required.

## Reuse Plan

The redesign should remain mostly contained to `src/tpa_analyzer/ui/app.py`.

### Logic to Keep

- analysis execution based on assigned group strings
- statistics execution using `group_order`
- plot export and styling behavior
- group color handling based on the managed group list
- existing result reordering logic tied to `group_order`

### Logic to Change

- stop initializing groups with `infer_group_from_filename`
- stop deriving new groups from file assignment handlers
- remove batch filename-term assignment handlers and related UI
- shift group creation/deletion/rename into explicit handlers

### Cleanup Rule

If `infer_group_from_filename` has no remaining callers after this redesign, remove it and any tests that exist only for that behavior.

## Edge Cases

- If no groups exist, files remain `UNASSIGNED` and group assignment actions stay disabled.
- If the active group is deleted, its files become unassigned and the active group selection clears or moves to the next available group.
- Renaming a group updates all files currently assigned to that group.
- Assigning a selected set of files overwrites their previous assignments.
- Session restore should preserve managed groups and file assignments even if no files are currently selected.

## Testing Strategy

Focus tests on behavior that changed while preserving downstream compatibility.

### Grouping Workflow Tests

- refreshing a directory loads files as unassigned
- creating, renaming, deleting, and reordering groups updates `group_order` correctly
- assigning multiple selected files updates all targeted file records
- clearing assignments returns files to the unassigned state
- deleting a group moves affected files to the unassigned state

### Compatibility Tests

- analysis still runs correctly with assigned groups
- exports and plots still honor the ordered group list
- color selection still tracks the managed group set

### Cleanup Verification

- verify whether `infer_group_from_filename` remains referenced anywhere
- if not referenced, remove the function and any now-obsolete tests

## Implementation Boundaries

This work should stay focused on the grouping workflow and targeted cleanup only. It should not expand into unrelated refactoring unless a local change is required to keep responsibilities clear inside `src/tpa_analyzer/ui/app.py`.
