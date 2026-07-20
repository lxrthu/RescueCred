# ToolSandbox V4.6 selective causal residual correction

V4.5 passed development by 0.83 points but failed frozen confirmation by
2.44 points. In both splits it shifted both classes toward candidate A; the
class-conditional separation did not replicate. The data and integrity gates
passed, so V4.6 changes only the learner.

Both arms start from the exact frozen V4.5 Mask adapter and use every one of
the 126 V4.4 training pairs once per epoch for three epochs in the same order.
The matched control continues to push B above A. V4.6 first caches the Mask
margin m0=logp(B)-logp(A). If the causal label is wrong or has signed margin
below 0.05, it requires a signed residual improvement of 0.05. Otherwise it
preserves the Mask margin with a squared retention loss. This makes causal
credit an error-correcting residual rather than a replacement policy.

The known offset-125 and offset-165 sets are development diagnostics only for
V4.6 because their outcomes informed this design. No fresh-task claim is made.
The development gate requires improvement over both frozen Mask and the
matched continued-Mask control, Rescue noninferiority, Reverse improvement,
nonnegative mean Rescue shift, mean Reverse shift at most -0.02, and more
wins than losses. A later protocol must freeze a new scenario profile before
any confirmatory claim.
