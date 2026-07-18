# ToolSandbox V4 Worker-Timeout Sanity Erratum

## Observed sanity result

The repaired V4 sanity completed one controlled and one natural paired event.
Both were progress-based rescues, terminal scores tied, and all snapshot/prefix
checks were exact. The run then stopped because proposal 2 exceeded the
180-second worker deadline. Terminating that worker caused proposal 3 to fail
immediately with `RuntimeError`, yielding a reported worker failure rate of 2/3.

No fresh offset-40 scenario or primary V4 holdout outcome was evaluated.

## Root cause

The persistent worker was intentionally terminated after a timeout but was not
restarted. Thus one provider stall deterministically cascaded into every later
request. The 180-second outer deadline was also shorter than an observed proxy
request for the frozen DeepSeek V4 Pro model.

## Frozen correction

- Set the outer per-request deadline to 600 seconds.
- Bind that deadline in the pre-outcome protocol lock.
- Restart the stateless worker after timeout or an observed prior exit.
- Count the original timed-out event as invalid; never retry it or manufacture
  an outcome. Restart applies only to the next independent request.

This is an execution-integrity repair based only on development sanity logs. It
does not change V4 credit components, thresholds, model, event identities,
horizons, or the fresh holdout.
