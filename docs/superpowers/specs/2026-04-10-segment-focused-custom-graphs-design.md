# Segment-Focused Custom Graphs Design

Date: 2026-04-10

## Summary

Refocus the custom graph feature around semantic TPA segments rather than treating overlays as the main organizing concept. A user should be able to choose a semantic region such as `B1 start -> Peak1` and make that region the main displayed graph domain. The graph should then show only that segment, with X rebased to zero, while still allowing a left-axis curve, an optional right-axis curve, and segment-relevant annotations.

After that segment-first flow exists, extend the same builder so users can choose whether the graph shows grouped data or selected individual samples. Sample selection should happen inside the Plot Builder in a dedicated list, using the same space-bar toggle interaction style already used elsewhere in the app, but without reusing the grouping UI.

## Goals

- Make semantic segments first-class graph domains rather than only overlay decorations.
- Rebase segment-focused graphs so the selected segment starts at `0` on the X axis.
- Preserve dual-axis plotting within a selected segment.
- Show derived values such as hardness and adhesiveness as annotations only.
- Add a dedicated sample-selection workflow inside Plot Builder.
- Support grouped and selected-sample graph scopes.
- Support two selected-sample display modes:
  - `stacked`: multiple chosen samples shown together as aligned stacked panels
  - `individual`: one graph per chosen sample
- Make event-loop safety an explicit design requirement in Textual.

## Non-Goals

- Arbitrary numeric slicing of curves.
- User-defined custom segment boundaries.
- Derived-value bars or inset summary blocks in segment graphs.
- Mixing grouped and selected-sample data in the same recipe.
- Reusing the grouping management UI for sample selection.

## Product Direction

The custom graph feature should evolve from “full-curve plotting with optional overlays” into a segment-first graph builder.

The user’s first conceptual choice is what part of the experiment the graph is about:

- the full curve
- a semantic segment such as:
  - `B1 start -> Peak1`
  - `Peak1 -> B1 end`
  - `B1 end -> B2 start`
  - `B2 start -> Peak2`
  - `Peak2 -> B2 end`
  - `Modulus window`

When a semantic segment is selected, the graph should:

- display only that segment
- rebase X so the segment begins at zero
- allow one left-axis curve and one optional right-axis curve inside the segment
- offer only annotations that make sense for that segment

This segment-focused mode becomes the primary custom-graph workflow. Sample granularity is layered on top of that, not solved before it.

## Builder Model And User Flow

The builder should no longer use overlay choice as the main organizing control. Instead, the first control should be `View Domain`:

- `Full Curve`
- `Semantic Segment`

If the user selects `Semantic Segment`, the builder shows a `Segment` control with the supported semantic regions.

The full graph-building flow should be:

1. choose `Full Curve` or `Semantic Segment`
2. choose the segment if semantic mode is active
3. choose the left-axis primary curve
4. choose the optional right-axis curve
5. choose segment-relevant annotations
6. choose data scope:
   - `Grouped`
   - `Selected Samples`
7. if `Selected Samples` is active, choose one or more samples from a dedicated in-builder selection list
8. if multiple samples are selected, choose display mode:
   - `Stacked`
   - `Individual`

Definitions for selected-sample display:

- `Stacked`: multiple selected samples are shown together in a single figure as aligned stacked panels.
- `Individual`: one graph is produced per selected sample.

This keeps the graph builder conceptually clean:

- what part of the experiment is shown
- which curves are shown inside that part
- which annotations interpret that part
- whether the graph represents groups or specific samples

## Textual UI Structure

The Plot Builder tab should contain a single structured builder with these sections:

- `View Domain`
- `Segment`
  - visible only in semantic-segment mode
- `Curves`
  - left-axis primary curve
  - optional right-axis curve
- `Annotations`
  - only values relevant to the chosen segment
- `Data Scope`
  - grouped or selected samples
- `Sample Selection`
  - visible only when `Selected Samples` is active
  - dedicated list inside Plot Builder
  - space toggles sample selection
- `Display Mode`
  - for selected samples: `Stacked` or `Individual`
- `Live Summary`
  - plain-language summary of the graph recipe

The sample-selection list must be separate from grouping management. It should not change assignments, group order, or grouping state. It is a plotting-state control only.

This is feasible in Textual with the current interaction style:

- dedicated in-tab selectable list
- space-bar toggling
- selection count/status summary
- guarded widget enable/disable and option refresh

## Data And Rendering Model

The internal custom graph recipe should be extended so segment focus and sample scope are explicit properties.

A saved custom graph recipe should include at least:

- title
- view domain: `full_curve` or `semantic_segment`
- semantic segment key when relevant
- x-axis mode
  - for this scope, semantic segments always use rebased mode
- left-axis primary curve
- optional right-axis curve
- annotation keys
- data scope: `grouped` or `selected_samples`
- selected sample filenames when relevant
- selected-sample display mode: `stacked` or `individual`
- existing curve/band style settings where still relevant

Rendering behavior should support:

- `Grouped + full curve`
  - current grouped custom graph behavior
- `Grouped + semantic segment`
  - crop traces to the chosen segment, rebase X to segment start, then render grouped means/curves within that segment
- `Selected samples + semantic segment + stacked`
  - one figure with aligned stacked panels, one selected sample per panel
- `Selected samples + semantic segment + individual`
  - one exported graph per selected sample

Segment-aware annotations must be constrained by meaning:

- hardness belongs on `B1 start -> Peak1`
- adhesiveness belongs on `B1 end -> B2 start`
- modulus annotation belongs on `Modulus window`

Annotations should be offered only when they make sense for the chosen segment.

## Error Handling

The builder should proactively prevent invalid graph recipes.

Builder behavior:

- if the user selects `Semantic Segment`, only segment-relevant annotations are offered
- if the user selects `Selected Samples`, sample selection becomes required before save
- if multiple samples are selected, `Stacked` becomes the default display mode
- if no QC markers exist for the selected segment, the graph should be unavailable rather than added and later failing at export

Rendering/export behavior:

- grouped and selected-sample segment modes must share the same extraction and rebasing logic
- if one selected sample lacks valid QC markers for the chosen segment, that sample should be skipped with a precise warning
- `Individual` mode filenames should include the sample name
- `Stacked` mode output should clearly indicate the segment and rebased-X interpretation

## Event Safety Requirement

Event-loop safety must be an explicit acceptance criterion for this feature.

The custom graph builder must follow these rules:

- user input handlers may request a builder refresh
- refresh logic may update widgets
- widget updates performed during refresh must not recursively trigger the same refresh/autosave path

Required safeguards:

- one dedicated “builder sync in progress” guard for the whole custom graph builder
- generic `Select.Changed` and `Checkbox.Changed` autosave handlers must ignore builder-internal updates while that guard is active
- helper methods that repopulate widget state must no-op when the state is already identical
- segment changes, sample toggles, annotation changes, and display-mode changes must use the same guarded refresh path

This is a hard requirement because the previous overlay feature regressed into an event-loop/autosave storm when builder state changes triggered repeated internal widget updates.

## Testing

Testing should cover:

- semantic segment extraction and X rebasing correctness
- segment-aware annotation availability
- sample-selection list behavior with space-bar toggling
- grouped vs selected-sample rendering
- `stacked` vs `individual` export behavior
- graceful skip/warning behavior when selected samples lack required QC markers
- event-loop safety:
  - one user action causes one state refresh and one autosave
  - changing segment
  - changing data scope
  - toggling sample selection
  - changing display mode
  - enabling/disabling annotations

## Delivery Boundaries

Included in this revised feature direction:

- semantic segment as the main custom graph domain
- rebased segment rendering
- grouped mode as a first-class scope
- dedicated in-builder sample selection
- selected-sample display modes:
  - `stacked`
  - `individual`
- segment-aware annotations only
- event-loop safety as an explicit implementation requirement

Deferred beyond this pass:

- arbitrary numeric slicing
- editable custom segment boundaries
- derived-value bars or inset summaries
- mixed grouped and selected-sample content in one graph
- broader custom graph redesign beyond segment-first flow

## Recommendation

Implement the next custom graph iteration as a segment-first builder, not as an overlay-first extension. This gives the graph a clearer meaning, aligns better with TPA interpretation, and creates a coherent path for later grouped versus selected-sample viewing without overloading the grouping UI or repeating the event-loop problems seen in the last iteration.
