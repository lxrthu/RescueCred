# Route-A AppWorld Dev Evaluation Tracker

| Run | Split | Systems | Seed | Primary metric | Status |
|---|---|---|---:|---|---|
| AW3-SANITY | AppWorld dev, first 3 frozen events | Mask vs V2 | 42 | evaluator completes for both | packaged, pending server |
| AW3-FULL | AppWorld dev, up to 57 frozen events | Mask vs V2 | 42 | paired official requirement score | packaged, gated by sanity |

Pre-registered full gate: at least 20 paired valid events, identical event hash, reference-free adapter inputs, V2 task-success non-inferiority, and strictly higher paired mean official requirement score.

Runtime estimate: approximately 10-30 minutes on two H200 GPUs, depending on AppWorld evaluator and shared-server load.
