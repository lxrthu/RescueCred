# Route-A Seed-42 Preference Pair — local code review

- Scope: same-bank Mask vs RescueCredit-v2 offline preference pilot.
- Data boundary: training reads only the frozen public bank and dense Shadow-credit file; private audit labels are excluded.
- Fairness: both methods share base model, event split, optimizer family, epochs, and held-out validation events.
- V2 behavior: positive delta learns B>A, negative delta learns A>B, zero delta is skipped.
- Mask behavior: B>A for every training event, intentionally exposing harmful-correction failure on reverse-credit cases.
- Probability: canonical action completion scored by mean token log-prob.
- Runtime: one single-GPU process per method; reference margins are cached because the base policy is fixed.
- Evidence boundary: passing this gate authorizes AppWorld dev evaluation only; it is not a task-success claim.
