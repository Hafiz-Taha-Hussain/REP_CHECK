"""
Form-check thresholds, organized per camera view (front / side).

Why per-view: depth (knee angle) is a robust signal from either angle, but
knee alignment (valgus) is a FRONTAL-plane motion -- only reliably visible
from a front-on camera -- and back angle (forward lean) is a SAGITTAL-plane
motion -- only reliably visible from the side. Trying to measure either one
from the wrong camera plane produced unreliable results (tested directly
this session): front-view back-angle had huge, non-separating variance
across a 25-subject dataset; side-view knee-alignment is blocked by the
hip-width guard, since hips nearly stack in x from a side profile.

Calibration history (kept for context, not all still active):
- Originally calibrated from 2 personal clips (front view only). Looked
  clean on 2 examples but didn't generalize -- see below.
- Depth threshold tested against a 25-subject dataset's confirmed good vs
  bad reps; held up as a real, if imperfect, separator.
- Back-angle (front view) tested the same way and did NOT separate good/
  bad at all, and confirmed-good reps varied 8-90 degrees across subjects
  -- too much natural variance for one flat threshold. Disabled for front.
- Back-angle (side view) has NOT been tested against true bad-lean
  examples (this dataset's "bad" label mixes multiple unrelated error
  types, confirmed by direct visual inspection of 3 "bad" clips -- each
  showed a different mistake, not consistently excessive lean). The side
  threshold below is a loose outlier-flag derived from confirmed-GOOD
  side reps only (95th percentile), not a validated pass/fail boundary.
- Knee alignment (front view) similarly not validated against a true bad-
  alignment example -- one deliberate test clip was inconclusive due to
  confounded variables (depth mismatch, asymmetric execution).

Bottom line: every enabled check here is either evidence-based (depth) or
explicitly flagged as a provisional outlier-flag, not a precision-tested
rule. None of this is "solved" -- it's the most honest version buildable
from the data actually available.
"""

# --- Depth (both views) ---
# Good clips bottomed at 18-35 deg; bad clips only reached 82-88 deg on the
# personal dataset. Standard fitness guidance (~90 deg = parallel) does NOT
# separate these correctly -- fit to this data's actual gap, not textbook
# guidance. Held up reasonably against the 25-subject front dataset too.
#
# Manually loosened 60 -> 65 deg on request, NOT independently re-validated
# against new data. Still sits comfortably inside the observed good/bad
# gap above (good clips <=35, bad clips >=82), so this stays a safe change
# within the evidence already gathered -- but the specific choice of 65,
# unlike 60, isn't itself backed by a data point. If a concrete reason
# surfaces later (a real clip this was too strict on, more calibration
# data), replace this note with that evidence.
DEPTH_THRESHOLD_DEG = 65  # knee angle must drop below this at the bottom

# --- Back angle / forward lean ---
# Front view: disabled. Tested against 25 subjects, confirmed-good reps
# ranged 8-90 deg (median 36, 90th pct 73, 95th pct 85) -- too much natural
# inter-person variance for a flat threshold; also a sagittal-plane motion
# a front camera can't observe directly (only via foreshortening, which is
# noisy). A genuinely good stock-footage squat was incorrectly failed by
# the old flat 35 deg cutoff for exactly this reason.
#
# Side view: enabled, provisional. Side view measures lean far more
# consistently (coefficient of variation ~0.27 vs ~0.43-0.63 for every
# front-view variant tried, including a shin-relative formula that made
# things worse). Threshold is the 95th percentile of confirmed-GOOD side
# reps (~78 deg) -- an outlier flag, not a tested good/bad boundary.
BACK_ANGLE_MAX_DEG_SIDE = 75

# --- Knee alignment (valgus) ---
# Front view: enabled, provisional. Calibrated from good-form clips only
# (their observed range becomes the "acceptable" band). Not validated
# against a true bad-alignment example -- treat as a loose first pass.
#
# Side view: disabled. This is a frontal-plane motion, not visible from
# the side -- and structurally, the hip-width the ratio divides by nearly
# vanishes in a side profile (hips stack in x), so the existing near-zero
# guard would reject nearly every frame anyway. Not worth scoring.
KNEE_ALIGNMENT_MIN = -1.85  # below this = flag as caving in beyond observed good range
KNEE_ALIGNMENT_MAX = 0.85   # above this = flag as tracking outward beyond observed good range
# (values widened slightly from console-rounded -1.8/0.8 -- the exact
# unrounded max in the defining clip was 0.837, which the rounded boundary
# incorrectly failed. Always derive thresholds from exact values.)

# --- Per-view check configuration ---
# Which checks run for a given camera view. Depth always runs; the other
# two are mutually exclusive by view, per the plane-of-motion reasoning
# above.
VIEWS = {
    "front": {
        "depth_enabled": True,
        "back_angle_enabled": False,
        "knee_alignment_enabled": True,
    },
    "side": {
        "depth_enabled": True,
        "back_angle_enabled": True,
        "knee_alignment_enabled": False,
    },
}