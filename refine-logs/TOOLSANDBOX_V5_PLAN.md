# ToolSandbox V5 independent causal router

V4.5 changed a shared LoRA policy and learned a global anti-B shortcut. V4.6
retained the frozen Mask behavior but produced no causal-accuracy improvement:
62 of 126 unique training events were routed to correction, yet development
selection changed only twice and Reverse margins moved in the wrong direction.

V5 therefore freezes the base model and the exact V4.5 Mask adapter. It caches
two public-only features for every training pair: a margin-only control vector
and a semantic vector made from a deterministic projection of the difference
between frozen completion representations plus the Mask margin. The supervised
target is not a new action preference; it is whether the frozen Mask decision
must be flipped to match the train-only Shadow direction.

Both routers are logistic heads trained on identical events. A deterministic
five-fold task-grouped out-of-fold procedure chooses a conservative flip
threshold using only frozen training labels. At deployment, the head may only
KEEP or FLIP the frozen Mask selection; it cannot update or generate actions.
The margin-only head controls for confidence calibration without semantic
features.

The offset-125 and offset-165 V4.5 evaluation sets are already known. They are
development and post-hoc diagnostics only. A passing development gate requires
the semantic router to beat both frozen Mask and the margin control, preserve
Rescue accuracy, improve Reverse accuracy, preserve terminal/progress metrics,
and produce more wins than losses. A new scenario profile must be frozen before
any confirmatory claim.
