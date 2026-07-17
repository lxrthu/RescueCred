# Route-A Bounded-Horizon Tracker

- Status: implementation and three-round fresh-agent code review complete (`DEPLOY YES`); remote sanity/full run pending.
- Frozen inputs: seed-42 dev events and frozen Mask/V2 selection files.
- Sanity: first 3 events, horizons 4 and 8.
- Full: all 55 events, primary horizon 8.
- Backend: AppWorld CPU plus Azure GPT-4o continuation; no GPU.
- Output: `outputs/route_a_appworld_dev_bounded_seed42/`.
- Stop rule: failed primary gate means no seed expansion under this protocol.
- Local validation: 10 focused tests passed; full repository regression passed; 20-file embedded bundle inventory verified.
- Deployment artifact: `dist/PASTE_ROUTE_A_BOUNDED_TO_SERVER.sh` (about 39 KB).
