# Route-A AppWorld Dev Evaluation V2 Code Review [local-only]

## Trigger

The first server evaluation had 50.9% generation failures and 0% exact corrections for both adapters, but the reference suffix still produced 57/57 official successes. That protocol had a hard ceiling and could not distinguish the methods.

## V2 fixes

- The deployable visible-candidate Harness constructs one shared candidate B without reference values.
- Mask and V2 adapters are used only for the operation they were trained for: mean-token-log-prob scoring of A versus B.
- After the selected action, a shared Azure GPT-4o temperature-0 continuation receives only visible instruction, schemas, receipts, and branch history.
- No reference suffix is executed. Reference actions before the event reconstruct controlled state only.
- Candidate and continuation failures remain scored; only a broken reference prefix invalidates an event.
- The paired gate uses the intersection of valid event IDs, requires no reference suffix, and requires operational adapter scoring.
- Only AppWorld `dev` is loaded; test splits remain untouched.

## Verdict

No local blocking issue remains. The three-event server GPU/API sanity must pass before the full paired dev run starts. Evidence remains a controlled-state task evaluation, not a fully autonomous reset-to-finish agent result.
