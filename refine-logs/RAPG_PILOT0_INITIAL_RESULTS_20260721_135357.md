# RAPG Surrogate Preflight Initial Results

## Local engineering sanity

- Python compile: passed.
- Ruff: passed.
- Existing estimator/audit plus RAPG tests: `10 passed, 2 skipped`.
- The two skips are the new Torch numerical tests because the Windows local environment has no Torch.
- Independent code review: DEPLOY YES for the surrogate preflight.

## Research result

Not available locally. Real LoRA score extraction, 1,000 audit resamples, and the independently recomputed gate must run on the server.

## Claim boundary

No positive RAPG or policy claim is supported. A passing surrogate gate only authorizes collection of a clean on-policy autoregressive bank.
