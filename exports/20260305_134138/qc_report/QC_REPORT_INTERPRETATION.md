# QC Report Interpretation Guide

This guide explains how to read the QC report outputs:
- `qc_summary.csv`
- `qc_control_parameters.csv`
- `qc_markers_and_areas.csv`
- `files/*_qc.png`

## 1) What the QC figure shows

Each `*_qc.png` has two panels:

1. **QC Force Map: Time vs Force Corrected**
- Black curve: baseline-corrected force.
- Dotted cyan line: trigger force used to detect cycle boundaries.
- Markers:
  - `B1 start`, `Peak1`, `B1 end` for first compression
  - `B2 start`, `Peak2`, `B2 end` for second compression
- Filled regions:
  - `A1` (blue): positive area in cycle 1
  - `A2` (green): positive area in cycle 2
  - `Adhesiveness` (red): negative area between cycles

2. **Modulus Context (True Stress vs True Strain)**
- Brown curve: measured true stress-true strain response.
- Yellow band: selected modulus strain window (`Modulus Strain Min/Max (%)`).
- Dashed dark line: linear fit in the selected window.
  - Slope of this line is the reported **Modulus (kPa)**.

## 2) How to interpret Modulus Context (most important)

Use these checks in order:

1. **Window placement**
- The yellow window should sit in the early, mostly linear loading segment.
- If it starts too early, baseline/trigger noise can dominate.
- If it extends too far, nonlinearity can bias slope.

2. **Linearity inside window**
- Dashed fit line should follow the curve closely through most of the window.
- Large curvature or visible mismatch means modulus is not representative for that window.

3. **Sufficient points**
- `Modulus Fit Points` in `qc_summary.csv` should not be very low.
- Too few points means unstable slope; adjust trigger/baseline/window.

4. **Physical consistency across replicates**
- Modulus values should vary reasonably within a group.
- Large random swings usually indicate poor window placement, noisy baseline, or inconsistent cycle detection.

## 3) Which controls to adjust first

Recommended tuning order:

1. **Baseline Points**
- Fixes offset drift before anything else.

2. **Trigger Force (N)**
- Controls cycle start/end boundaries.
- Too low: includes noise tails.
- Too high: trims valid low-force region.

3. **Peak Prominence (N)** and **Peak Distance (pts)**
- Stabilize `Peak1/Peak2` detection when data are noisy or closely spaced.

4. **Modulus Strain Min/Max (%)**
- Final refinement of modulus window after cycle detection is stable.

## 4) Quick symptom -> action table

- **Fit line obviously curved mismatch**
  - Narrow the modulus window and move it earlier.
- **Very low Modulus Fit Points**
  - Lower trigger slightly or widen modulus window moderately.
- **Peaks misplaced (wrong cycle picked)**
  - Increase peak prominence and/or peak distance.
- **A1/A2 areas look truncated**
  - Trigger likely too high.
- **Adhesiveness area missing when expected**
  - Check cycle boundary overlap; reduce trigger or review data quality.

## 5) Notes on metrics linked to QC regions

- **Hardness** = force at `Peak1`.
- **Cohesiveness** = `A2 / A1`.
- **Resilience** = `A1_down / A1_up`.
- **Springiness** = ratio of time-to-peak in cycle 2 vs cycle 1.
- **Chewiness** = Hardness x Cohesiveness x Springiness.

When QC regions are wrong, derived metrics are wrong even if formulas are correct.
