from rescuecredit.deltaguard_evaluation import evaluate_deltaguard


def test_separate_probe_auc_and_whole_stream_metrics():
    source = []
    probes = []
    labels = {}
    baselines = {}
    for family in ("messaging", "settings"):
        for label, score in ((0, 0.0), (1, 1.0)):
            event_id = f"{family}-{label}"
            source.append(
                {
                    "event_id": event_id,
                    "task_id_hash": event_id,
                    "family": family,
                    "selected": True,
                }
            )
            probes.append(
                {
                    "event_id": event_id,
                    "family": family,
                    "reverse_score": score,
                    "contract_reverse_score": score,
                }
            )
            labels[event_id] = label
            baselines[event_id] = 0.5
    result = evaluate_deltaguard(
        source_rows=source,
        probe_rows=probes,
        labels=labels,
        baseline_scores=baselines,
        min_class_per_family=1,
        min_auc=0.75,
        min_auc_gain=0.10,
        max_probe_rate=1.0,
    )
    assert result["conditional_discriminability"]["typed_delta_roc_auc"] == 1.0
    assert result["conditional_discriminability"]["v7_receipt_roc_auc"] == 0.5
    assert result["whole_stream_public_paired_deltas"]["reverse_recall"] == 1.0
    assert result["whole_stream_public_paired_deltas"]["rescue_drop"] == 0.0
    assert result["feasibility_passed"] is True


def test_fixed_cohort_returns_inconclusive_without_class_coverage():
    source = [{"event_id": "x", "task_id_hash": "t", "family": "settings", "selected": True}]
    probes = [{"event_id": "x", "family": "settings", "reverse_score": 0.5, "contract_reverse_score": 0.5}]
    try:
        evaluate_deltaguard(
            source_rows=source,
            probe_rows=probes,
            labels={"x": 0},
            baseline_scores={"x": 0.5},
            min_class_per_family=1,
            min_auc=0.75,
            min_auc_gain=0.10,
            max_probe_rate=1.0,
        )
    except ValueError as error:
        # Whole-stream metrics correctly reject a one-class source stream rather
        # than manufacturing a routing result.
        assert "Rescue and Reverse" in str(error)
    else:
        raise AssertionError("single-class stream should not produce routing metrics")
