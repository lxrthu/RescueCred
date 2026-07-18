# ToolSandbox V4 Scenario-Pool Preflight Erratum

## Trigger

The first cloud preflight stopped before protocol freezing with:

```text
RuntimeError: insufficient fresh ToolSandbox scenarios at offset 40: 0 < 30
```

No V4 sanity, holdout, branch score, or outcome was produced.

## Root cause

`select_scenarios` required `NO_DISTRACTION_TOOLS`. That eligible universe had
exactly 40 rows. The protocol then requested a fresh slice beginning at offset
40, which is necessarily empty. The Stage-0 report's 1032 total scenarios did
not imply 80 rows under this narrower filter.

## Frozen correction

V4 uses a deterministic tiered pool:

1. no-distraction state-dependency scenarios;
2. no-distraction other eligible scenarios;
3. distraction-tool state-dependency scenarios;
4. distraction-tool other eligible scenarios.

All tiers still require single-user, multiple-tool scenarios and exclude the
RapidAPI search tools. Each tier is shuffled deterministically with the frozen
seed. The first two tiers preserve the old selection as an exact prefix, so
offset 40 draws only previously unused scenarios.

## Integrity statement

This is a pre-outcome partition repair. It changes neither the V4 credit rule,
horizons, thresholds, model, nor outcome gates. The protocol lock records the
new pool profile and exact fresh scenario hashes before any branch is scored.
