# Route-A AppWorld Dev Evaluation Code Review [local-only]

## Verdict

No blocking issue remains in the local implementation. Server-side three-event GPU sanity is still required before the full dev pass.

## Checklist

- Method fidelity: both methods use their already-frozen seed-42 adapters; no further optimization occurs during dev evaluation.
- Fairness: Mask and V2 receive one immutable dev event file, identical prompt contract, seed, reference fixture, suffix, and official evaluator.
- Leakage boundary: adapter workers receive only the serialized public prompt. Protected reference actions, suffixes, and evaluator results remain inside the AppWorld evaluator process.
- Ground truth: the primary metric is AppWorld's official requirement pass fraction, not another model's answer. Exact reference-action match is diagnostic only.
- Pairing: the gate computes means and success rates on the intersection of valid event IDs, preventing unequal-denominator comparisons.
- Failure handling: malformed generation falls back to the visible proposal and remains a model failure; candidate/suffix execution errors are scored rather than silently excluded. Only broken reference-prefix reconstruction invalidates an event.
- Split boundary: `dev` only; test splits are not loaded.
- Reproducibility: seed, event hash, adapter path, per-event result, and parseable summaries are recorded.
- Evidence scope: controlled-state dev evaluation, not fully autonomous end-to-end AppWorld agent performance.
