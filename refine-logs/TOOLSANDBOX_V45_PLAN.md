# ToolSandbox V4.5 matched anchored learner protocol

Date: 2026-07-20

V4.4 passed its frozen data gate with 126 nonzero, schema-valid candidate
pairs: 41 rescue and 85 reverse preferences spanning 21 and 36 tasks. V4.5
tests whether causal direction labels improve selection over a matched Mask
learner on the same candidate pool.

Before any V4.5 development outcome is observed, freeze the learner, the
offset-125 development candidate protocol, and the offset-165 confirmation
candidate protocol. Both evaluation sets use the same reference-free,
unranked, schema-valid candidate construction as V4.4. Offset 165 is not run
unless development passes.

Mask and V4.5 receive the same 126-event pool, deterministic balanced event
sequence, 126 presentations per epoch for three epochs, optimizer budget,
DPO-shift plus absolute-margin objective, and reference anchor. Mask always
labels B over A; V4.5 follows the frozen Shadow direction. Official outcomes
never enter prompts and join only after adapter scoring.

Development requires at least 40 events, five examples of each direction,
three selection disagreements, positive causal-accuracy improvement, more
wins than losses, terminal and progress noninferiority, and a class-conditional
margin-shift gap of at least 0.02. Confirmation additionally requires at least
0.05 causal-accuracy improvement. These are controlled-state preference
diagnostics, not autonomous task-success claims.
