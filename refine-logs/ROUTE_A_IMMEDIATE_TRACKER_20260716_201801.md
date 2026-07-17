# Route-A Immediate Experiment Tracker

- Status: implementation complete; remote run pending.
- Input: frozen seed-42 AppWorld dev events plus frozen Mask/V2 selections.
- Experiment: deterministic immediate official scoring of A and B after the same prefix.
- Sanity gate: 3 events must produce `immediate_summary.json`.
- Full gate: fixed in `rescuecredit/route_a_immediate.py` and documented in `docs/ROUTE_A_APPWORLD_IMMEDIATE_CN.md`.
- Output: `outputs/route_a_appworld_dev_immediate_seed42/`.
- Claim boundary: immediate action-selection evidence only, not end-to-end task success.
