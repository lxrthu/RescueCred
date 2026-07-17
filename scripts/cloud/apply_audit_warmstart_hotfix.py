#!/usr/bin/env python3
"""Apply the RescueCredit audit warm-start hotfix to an uploaded project."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    if old not in text:
        raise RuntimeError(f"cannot find expected block in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_audit() -> bool:
    path = ROOT / "rescuecredit" / "audit.py"
    old = '''class UniformAuditScheduler:
    def __init__(self, probability: float) -> None:
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        self.probability = float(probability)

    def probability_for(self, _event: object, _mu: float) -> float:
        return self.probability
'''
    new = '''class UniformAuditScheduler:
    """Uniform auditing with an exact per-patch warm start."""

    def __init__(self, probability: float, warm_start_events_per_patch: int = 0) -> None:
        if not 0.0 < probability <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        if warm_start_events_per_patch < 0:
            raise ValueError("warm_start_events_per_patch must be non-negative")
        self.probability = float(probability)
        self.warm_start_events_per_patch = int(warm_start_events_per_patch)
        self._seen_by_patch: dict[str, int] = {}
        self.warm_start_assignments = 0

    def probability_for(self, event: object, _mu: float) -> float:
        patch_id = getattr(event, "patch_id", None)
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError("audit event must expose a non-empty patch_id")
        seen = self._seen_by_patch.get(patch_id, 0)
        self._seen_by_patch[patch_id] = seen + 1
        if seen < self.warm_start_events_per_patch:
            self.warm_start_assignments += 1
            return 1.0
        return self.probability
'''
    return replace_once(path, old, new, "warm_start_events_per_patch")


def patch_engine() -> bool:
    path = ROOT / "rescuecredit" / "engine.py"
    changed = False
    changed |= replace_once(
        path,
        "        self.exact_confidence_threshold = exact_confidence_threshold\n",
        "        self.exact_confidence_threshold = exact_confidence_threshold\n"
        "        self.eligible_events = 0\n"
        "        self.audited_events = 0\n"
        "        self.valid_audits = 0\n",
        "self.eligible_events = 0",
    )
    changed |= replace_once(
        path,
        "        if not event.shadow_safe:\n            return EstimateOutcome(event, audited=False, identifiable=False)\n",
        "        if not event.shadow_safe:\n            return EstimateOutcome(event, audited=False, identifiable=False)\n"
        "        self.eligible_events += 1\n",
        "self.eligible_events += 1",
    )
    changed |= replace_once(
        path,
        "        if draw.draw:\n            if shadow is None and shadow_factory is not None:\n",
        "        if draw.draw:\n            self.audited_events += 1\n"
        "            if shadow is None and shadow_factory is not None:\n",
        "self.audited_events += 1",
    )
    changed |= replace_once(
        path,
        "            self.budget.charge_shadow(shadow.steps)\n            event.shadow_return = shadow.return_value\n",
        "            self.budget.charge_shadow(shadow.steps)\n"
        "            self.valid_audits += 1\n"
        "            event.shadow_return = shadow.return_value\n",
        "self.valid_audits += 1",
    )
    return changed


def patch_train() -> bool:
    path = ROOT / "scripts" / "run_train.py"
    changed = False
    changed |= replace_once(
        path,
        '    parser.add_argument("--audit-probability", type=float, default=0.2)\n',
        '    parser.add_argument("--audit-probability", type=float, default=0.2)\n'
        '    parser.add_argument("--audit-warm-start-events", type=int, default=2)\n',
        'parser.add_argument("--audit-warm-start-events"',
    )
    changed |= replace_once(
        path,
        "        UniformAuditScheduler(args.audit_probability),\n",
        "        UniformAuditScheduler(\n"
        "            args.audit_probability,\n"
        "            warm_start_events_per_patch=args.audit_warm_start_events,\n"
        "        ),\n",
        "warm_start_events_per_patch=args.audit_warm_start_events",
    )
    audit_reduce = '''    audit_stats = accelerator.reduce(
        torch.tensor(
            [
                engine.eligible_events,
                engine.audited_events,
                engine.valid_audits,
                engine.scheduler.warm_start_assignments,
            ],
            dtype=torch.long,
            device=accelerator.device,
        ),
        reduction="sum",
    )
    accelerator.wait_for_everyone()
'''
    changed |= replace_once(
        path,
        "    accelerator.wait_for_everyone()\n",
        audit_reduce,
        "audit_stats = accelerator.reduce(",
    )
    audit_summary = '''                "audit_stats": {
                    "eligible_events": int(audit_stats[0].item()),
                    "audited_events": int(audit_stats[1].item()),
                    "valid_audits": int(audit_stats[2].item()),
                    "warm_start_assignments": int(audit_stats[3].item()),
                },
'''
    changed |= replace_once(
        path,
        '                "budget_unused": args.total_interaction_budget - int(global_counts.sum().item()),\n',
        '                "budget_unused": args.total_interaction_budget - int(global_counts.sum().item()),\n'
        + audit_summary,
        '"audit_stats": {',
    )
    changed |= replace_once(
        path,
        '                    "audit_probability": args.audit_probability,\n',
        '                    "audit_probability": args.audit_probability,\n'
        '                    "audit_warm_start_events": args.audit_warm_start_events,\n',
        '"audit_warm_start_events": args.audit_warm_start_events',
    )
    return changed


def patch_gate() -> bool:
    path = ROOT / "scripts" / "check_pilot_gate.py"
    changed = False
    changed |= replace_once(
        path,
        '    parser.add_argument("--rescue", type=Path, required=True)\n',
        '    parser.add_argument("--rescue", type=Path, required=True)\n'
        '    parser.add_argument("--rescue-run-summary", type=Path)\n'
        '    parser.add_argument("--min-audited-events", type=int, default=0)\n',
        'parser.add_argument("--rescue-run-summary"',
    )
    old = '''    passed = any(value > 0 for value in improvements.values())
    result = {"passed": passed, "rule": "expand only if RescueCredit improves S_off or First-pass in this pilot", "improvements": improvements}
'''
    new = '''    if args.min_audited_events < 0:
        raise SystemExit("--min-audited-events must be non-negative")
    if args.min_audited_events and args.rescue_run_summary is None:
        raise SystemExit("--rescue-run-summary is required when --min-audited-events is positive")
    audited_events = None
    audit_gate_pass = True
    if args.rescue_run_summary is not None:
        run_summary = json.loads(args.rescue_run_summary.read_text(encoding="utf-8"))
        audited_events = int(run_summary.get("audit_stats", {}).get("valid_audits", 0))
        audit_gate_pass = audited_events >= args.min_audited_events
    metric_gate_pass = any(value > 0 for value in improvements.values())
    passed = metric_gate_pass and audit_gate_pass
    result = {
        "passed": passed,
        "rule": "expand only if RescueCredit improves S_off or First-pass and meets the audit floor",
        "improvements": improvements,
        "metric_gate_pass": metric_gate_pass,
        "audit_gate": {
            "passed": audit_gate_pass,
            "valid_audits": audited_events,
            "minimum": args.min_audited_events,
        },
    }
'''
    changed |= replace_once(path, old, new, "metric_gate_pass = any")
    return changed


def main() -> None:
    changed = {
        "rescuecredit/audit.py": patch_audit(),
        "rescuecredit/engine.py": patch_engine(),
        "scripts/run_train.py": patch_train(),
        "scripts/check_pilot_gate.py": patch_gate(),
    }
    for name, was_changed in changed.items():
        print(f"{'PATCHED' if was_changed else 'ALREADY_PATCHED'} {name}")
    print("AUDIT_WARMSTART_HOTFIX_OK")


if __name__ == "__main__":
    main()
