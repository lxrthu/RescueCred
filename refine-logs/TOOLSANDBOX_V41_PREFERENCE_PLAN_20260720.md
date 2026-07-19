# ToolSandbox V4.1 Same-Data Preference Comparison

Status: frozen design before training or offset-125 evaluation outcomes
Date: 2026-07-20

This timestamped plan is identical in experimental meaning to
`TOOLSANDBOX_V41_PREFERENCE_PLAN.md`. It freezes the passed offset-85 V4.1 audit
as common training credit, an untouched offset-125 40-scenario H8 evaluation,
and a same-event/same-order/same-budget Mask versus V4 comparison. Models see
only visible history, relevant public schemas, and candidate actions. The
evaluation gate requires at least 20 nonzero pairs, two reverse pairs, three
selection disagreements, at least +5 percentage points causal accuracy, more
V4 wins than losses, and terminal/progress noninferiority. The scope is
controlled-state preference learning, not autonomous task success.
