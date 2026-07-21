# Goal-Directed ActiveShadow Pilot 0

## Question

Can a single event-specific, read-only query expose a hard precondition failure
of Harness action B that static schema checks alone miss?

## Method

The frozen selector sees only the user-visible history, public tool schemas,
and fixed A/B actions. It first checks public JSON-schema obligations. When B
references a contact/reminder or requires cellular service, it executes at most
one read-only query on the replayed prefix. It routes back to A only when A is
schema-valid and B has either a public-schema violation or a refuted hard
precondition. Unknown evidence abstains to B.

No A/B branch is executed during collection. Labels and the official evaluator
remain physically unavailable until evaluation.

## Pilot gate

- at least 12 events, including at least 3 Rescue and 3 Reverse;
- empirical Rescue drop at most 2%;
- Reverse recall at least 20%;
- at least one Reverse hit added specifically by a real query beyond schema;
- at most one query per event and no collection errors.

This is an applicability-conditioned feasibility pilot. Passing does not make
a formal risk or paper-facing claim; it only authorizes a fresh task-disjoint
confirmation with deployment-rate accounting.

The original engineering preflight requested 20 events. Before any label was
read, the public-only freezer found exactly 16 eligible events among 370 public
events. The minimum was therefore reduced to 12 and the per-class coverage
minimum to 3. This is a sample-availability repair only; outcome thresholds are
unchanged, all 16 eligible events are collected, and the pilot remains
non-confirmatory.
