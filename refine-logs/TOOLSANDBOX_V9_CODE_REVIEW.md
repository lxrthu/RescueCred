# ToolSandbox V9 code review

Status: provisional same-family review; blocking issues resolved.

The first review confirmed that V9 reads only receipt/action positions 1 and 2,
rejects protected outcome keys, ignores third-plus receipts and all official
branch metrics, and preserves raw/train/cache event binding.

The reviewer required stronger authentication of the V7 one-step baseline. V9
now binds the V7 protocol, run summary, and OOF files; requires exact status,
stage, configuration, gate definition, and successful integrity checks; pairs
the same 126 event/task identities; and recomputes the V7 AUC from bound OOF
predictions.

The review also clarified live cost accounting: each probed event attempts two
continuation-policy calls, while tool executions equal two plus the number of
branches that produce a second action, with four as the maximum.
