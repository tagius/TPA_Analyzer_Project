# Custom Graph Redesign

Date: 2026-04-09

## Summary

Redesign the custom graph feature so it supports flexible post-analysis exploration while preventing invalid or misleading graph combinations. The new feature should let users build mixed plots anchored to a shared raw-domain X axis, combine compatible trace curves on left and right Y axes, and optionally add one derived overlay such as a semantic segment highlight, modulus window, or metric annotation/bar summary.

The current checkbox-based builder remains too generic to express TPA-specific plotting intent. It validates only basic trace-vs-metric separation and does not model dual-axis composition, semantic segments, or post-analysis overlays. The redesign replaces that flat model with a compatibility-driven builder and a composition-oriented saved plot specification.

## Goals

- Allow broad but filtered graph exploration rather than a fixed recipe list.
- Anchor each custom graph to one continuous raw-domain X axis such as `Time (s)`, `Aligned Time (s)`, or `True Strain (%)`.
- Support up to two left-axis trace series, one optional right-axis trace series, and one optional derived overlay.
- Make derived values available only after analysis has run.
- Support fixed semantic curve regions based on existing QC markers.
- Prevent invalid combinations in the UI instead of relying on late export-time failures.
- Keep the interaction model simple and realistic for Textual.

## Non-Goals

- Arbitrary numeric slicing of curves.
- Unlimited layer stacking.
- Mixed categorical and continuous X domains in the same figure.
- Full in-app rendered preview inside Textual.
- Free-form generic X/Y plotting detached from TPA semantics.

## Product Shape

The feature should be reframed as a mixed-plot builder for TPA analysis results, not as a generic X/Y checkbox tool. Every saved custom graph is a coherent composed figure with a single continuous X anchor and up to three visual layer groups:

- Left-axis curve layers from compatible trace variables.
- One optional right-axis curve layer from a compatible but distinct trace variable.
- One optional derived overlay layer based on analysis or QC-derived information.

Supported examples in first scope include:

- `Time (s)` with `Force Corrected (N)` on the left axis and `Deformation (mm)` on the right axis.
- `True Strain (%)` with `True Stress (kPa)` and a modulus-window overlay.
- `Time (s)` with `Force Corrected (N)` and a `B1 start -> Peak1` highlight.
- `Time (s)` with `Force Corrected (N)` and an adhesiveness highlight.
- `Time (s)` with `Force Corrected (N)` and a hardness annotation at `Peak1`.

Derived overlays should be available only after analysis exists. Before analysis, the builder can still configure raw trace plots, but post-analysis overlays remain disabled with explanatory messaging.

## Architecture

The redesign should introduce three explicit layers of responsibility.

### 1. Builder state

The Textual UI owns transient selection state only:

- selected X domain
- selected left-axis trace layers
- selected right-axis trace layer
- selected derived overlay
- rendering options such as curve mode and band mode

The UI must never infer validity by trial and error. It should consume compatibility metadata and present only eligible options.

### 2. Saved plot recipe

The current flat `GraphSpec` should be replaced by a composition-oriented spec that describes plot intent directly. The exact field names can be decided during planning, but the model should capture:

- title
- enabled state
- anchor X domain
- left-axis layers
- optional right-axis layer
- optional derived overlay layer
- curve rendering options
- annotation and grouping options as needed

This saved recipe is the session-persisted contract between UI and renderer.

### 3. Render jobs

The plotting engine should expand one saved recipe into concrete rendering steps:

- render left-axis curves
- render optional right-axis curve
- render semantic overlays such as filled regions or windows
- render metric annotations or compact inset summary bars when requested

This explicit expansion is necessary because mixed plots such as hardness plus a semantic curve segment are compositions, not simple X/Y plots.

## Compatibility Registry

The current variable registry is too shallow. The redesign should add a richer compatibility registry where each selectable plot item declares:

- source kind: raw trace, semantic segment, derived metric, annotation
- allowed X domains
- allowed visual roles: left curve, right curve, highlight, annotation, inset bar
- unit family
- domain family
- whether it requires post-analysis results
- whether it requires QC markers

The UI should use this registry to populate options and compute eligibility dynamically.

Example rules for first scope:

- `Force (N)` and `Force Corrected (N)` may share the same X domain, but the UI should avoid offering them together by default because they are redundant.
- `Force Corrected (N)` can pair with `Deformation (mm)` on opposite axes.
- `True Stress (kPa)` should strongly prefer `True Strain (%)` as X.
- Hardness, adhesiveness, chewiness, springiness, and cohesiveness are not free curve layers. They are derived overlays or annotations only.
- `B1 start -> Peak1`, `Peak1 -> B1 end`, `B1 end -> B2 start`, `B2 start -> Peak2`, `Peak2 -> B2 end`, and modulus window are semantic overlays tied to trace plots.

## Textual UI Strategy

A single-screen form is the preferred UI pattern. Textual can support this cleanly using ordinary widgets and dynamic enablement rather than a heavy multiselect control.

The builder should contain five always-visible blocks:

- `X Domain`: one `Select` for the anchor domain.
- `Left Axis`: an `OptionList` or checkbox list of eligible trace layers, capped at two selections.
- `Right Axis`: one optional `Select` populated only with trace layers compatible with the current X domain and not already used on the left.
- `Derived Overlay`: one optional `Select` for semantic segment overlays, metric annotations, or compact inset bars valid for the current plot.
- `Live Summary`: a read-only summary describing the graph that will be saved and any non-blocking warnings.

Expected UI behavior:

- Derived overlays stay disabled until analysis results are available.
- Changing X recomputes all downstream eligibility.
- Selecting a left-axis layer removes incompatible right-axis and overlay choices.
- The UI prevents impossible combinations instead of surfacing most errors after `Add Graph`.
- The saved graph list shows recipe-style summaries rather than raw `x=... y=...` dumps.

This is intentionally not a desktop-style multiselect widget. A coordinated form is simpler, more robust in Textual, and better aligned with TPA-specific constraints.

## Rendering Behavior

Each custom graph remains anchored to a shared continuous X domain. Derived layers must respect that constraint.

For first scope:

- left-axis layers are trace curves
- right-axis layer is an optional trace curve
- derived overlay is one semantic segment, modulus window, annotation, or compact inset summary bar

When a derived metric is shown in a bar-like form, it should appear as a compact inset or anchored summary element within the figure rather than as a separate categorical X-axis chart. This preserves the shared raw-domain model and avoids misleading dual-domain figures.

Suggested high-value combinations for first delivery:

- `Time (s)` + left `Force Corrected (N)` + right `Deformation (mm)`
- `Time (s)` + left `Force Corrected (N)` + overlay `B1 start -> Peak1`
- `Time (s)` + left `Force Corrected (N)` + overlay `Peak1 -> B1 end`
- `Time (s)` + left `Force Corrected (N)` + overlay `B1 end -> B2 start`
- `Time (s)` + left `Force Corrected (N)` + annotation `Hardness (N)` at `Peak1`
- `Time (s)` + left `Force Corrected (N)` + annotation `Adhesiveness`
- `Time (s)` + left `Force Corrected (N)` + overlay `B2 start -> Peak2`
- `True Strain (%)` + left `True Stress (kPa)` + overlay `Modulus window`
- `Aligned Time (s)` + group mean `Force Corrected (N)` + group mean `Deformation (mm)`

## Error Handling

The design should favor proactive constraint and narrow fallback behavior.

- The builder should disable impossible combinations rather than allowing invalid input.
- The renderer should still validate saved recipes strictly for session migration, export, and future compatibility.
- If a saved recipe depends on analysis output or QC markers that are unavailable, that one graph should be skipped with a precise warning.
- Invalid custom graphs must not fail the entire export batch.

## Testing

Testing should be split across three layers.

### Compatibility logic

Add tests proving that the registry offers or rejects the right combinations for:

- X domain filtering
- left/right-axis compatibility
- derived overlay eligibility
- post-analysis gating
- semantic segment availability

### UI state transitions

Add Textual-facing tests proving that:

- changing X updates eligible options
- choosing left-axis layers constrains the right axis
- derived overlay controls enable only after analysis results exist
- the live summary reflects the composed recipe accurately

### Rendering

Add plotting tests proving that:

- composed recipes expand into the expected render steps
- right-axis rendering behaves correctly
- semantic segment overlays render only when marker data exists
- missing optional data yields warnings rather than hard failures

## Migration And Persistence

Session persistence must continue to work. Existing saved graph specs should migrate forward where practical.

- Legacy flat specs that map cleanly to a simple trace-only recipe should be normalized into the new structure.
- Specs that cannot map cleanly should be preserved only if a safe fallback exists; otherwise they should be skipped with a migration warning.
- The session layer should remain responsible for normalizing persisted graph payloads before UI or export code consumes them.

## Delivery Boundaries

Included in first implementation:

- new composed custom-graph spec
- compatibility registry for valid combinations
- single-screen Textual builder redesign
- post-analysis derived overlay support
- fixed semantic segments based on QC markers
- left axis, right axis, and one derived overlay layer
- session migration for older graph specs where feasible

Deferred beyond first implementation:

- arbitrary numeric slicing
- unlimited recipe/layer composition
- mixed categorical and continuous X domains in one figure
- embedded live image preview in Textual

## Recommendation

Use a compatibility-driven single-screen builder rather than a recipe-only template system or a free-form layer queue. It best matches the goal of flexible exploration with filtered validity, fits the current Textual app architecture, and provides a clear foundation for later extension without preserving the current non-functional checkbox model.
