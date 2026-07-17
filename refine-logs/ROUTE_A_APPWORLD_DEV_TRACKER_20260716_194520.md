# Route-A AppWorld Dev Evaluation Tracker V2

| Run | Protocol | Result / Status |
|---|---|---|
| AW3-V1 | free-form adapter generation + reference suffix | invalid diagnostic: 50.9% generation failure, 0% exact correction, both methods 1.0 official score |
| AW3-V2-SANITY | shared reference-free B + adapter A/B scoring + reference-free continuation, first 3 events | packaged, pending server |
| AW3-V2-FULL | same V2 protocol, all available dev events | packaged, gated by sanity |

V1 is retained as evidence of an evaluation ceiling and must not be used as a method result. V2 writes to `outputs/route_a_appworld_dev_seed42_v2` and never executes a reference suffix.
