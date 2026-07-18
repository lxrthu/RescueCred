# ToolSandbox V4 Lexicographic Regret Plan

Status: frozen design before V4 outcomes  
Date: 2026-07-18

## Motivation and development evidence

The frozen terminal-credit H8 audit produced 36 valid controlled events but
only one nonzero terminal outcome (2.78%) and one natural repair with zero
terminal delta. This is a negative result for terminal-only credit. It is used
to motivate V4 and must not be reused as confirmatory evidence.

## V4 credit contract

For two replay-valid branches B (corrected) and A (original), compare the
following measured components lexicographically. A later component is consulted
only when every earlier component ties within `1e-12`.

1. final official ToolSandbox `similarity`, B minus A;
2. mean official `similarity` over the bounded horizon, padding an early stop
   with its final state, B minus A;
3. visible tool-call errors, A minus B;
4. official evaluator turn count, A minus B;
5. executed branch actions, A minus B.

The first nonzero component determines `rescue_preference` or
`reverse_preference`. If every component ties, the event is `zero_delta`.
No weighted scalar reward is constructed. Task outcome, progress timing, errors,
and cost remain separately reportable. Efficiency credit must never be reported
as task-success improvement.

## Frozen evaluation

- ToolSandbox commit: `165848b9a78cead7ca7fe7c89c688b58e6501219`.
- Development sanity: offset 0, limit 3, seed 42, H4.
- Fresh primary audit: offset 40, limit 40, seed 42, H8.
- Event search: first eligible treatment along at most eight reference-free
  visible prefix actions.
- Continuation: DeepSeek V4 Pro, non-thinking, temperature 0, via the configured
  OpenAI-compatible proxy.
- No milestones, minefields, reference actions, hidden state, evaluator outputs,
  or reference suffix enter proposal, repair, or continuation workers.

If fewer than 30 fresh scenarios exist at offset 40, stop and report insufficient
fresh coverage; do not reuse the development scenarios.

## Frozen gates

Mechanism gate:

- at least 30 fresh selected scenarios;
- at least 20 valid controlled pairs;
- at least 20% nonzero lexicographic causal decisions;
- exact snapshot restoration and official evaluator provenance;
- worker failure rate at most 10%.

Deployable-Harness gate additionally requires at least three replay-valid natural
visible-error repair pairs, at least three nonzero natural decisions, at least two
natural rescues, at least one rescue based on official outcome/progress rather than
efficiency alone, wins greater than losses, and natural harm rate at most 10%.
A mechanism-only pass permits a controlled preference pilot but does not authorize
a deployable Harness or autonomous task-success claim.

The protocol lock must bind the complete source inventory, Stage-0 gate, pinned
ToolSandbox runtime, model/provider/base URL/thinking mode, plan hash, thresholds,
and disjoint development/fresh scenario hashes. Every intermediate official score
must be read-only, carry official provenance, and independently reproduce its
padded horizon AUC before any V4 gate can pass.

## Stop rule

If the fresh mechanism gate fails, stop V4 ToolSandbox work and move to the next
environment. Do not tune component order, tolerances, horizon, offset, or gates on
the fresh outcomes.
