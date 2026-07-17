# RescueCredit-v2 event-coverage recovery

Date: 2026-07-15

## Trigger

The untouched full-data deployable Seed-42 pilot was degenerate: Mask-v2 and
RescueCredit-v2 both reached 7/18 on Dev, while RescueCredit-v2 recorded zero
eligible events, zero audits, and zero shadow steps. The causal component was
therefore inactive.

## Shared sampling diagnostic

- Rebuild public tool catalogs from benchmark scenario filenames plus schema
  prerequisite closure, not from parsed `reference_actions`.
- Keep references only for offline coverage checks, environment reward, split
  construction, and evaluation.
- Use a shared 0.75 visible-structure curriculum for Mask-v2 and
  RescueCredit-v2; keep the original Dev evaluation unchanged.
- Match split hash, world size, main-step budget, visible-pool hash,
  curriculum fraction, and assignment-sequence hash across methods.

The defensible scope is **reference-action-field-independent intervention and
curriculum selection over benchmark-declared visible tool catalogs**. This is
not fully non-Oracle tool discovery.

## Sanity gate

Run `scripts/cloud/run_v2_visible_curriculum_smoke_2gpu.sh` for 512 main steps.
Proceed to a new 2000-step comparison only if it produces at least five
eligible events, five valid audits, three nonzero causal events, and zero
failed replays.
