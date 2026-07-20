from __future__ import annotations

from typing import Any, Mapping


def validate_contract(
    contract: Mapping[str, Any], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("version") != "pic-v2":
        errors.append("unsupported_version")
    hashes = contract.get("action_hashes")
    if not isinstance(hashes, Mapping):
        errors.append("missing_action_hashes")
    else:
        if hashes.get("A") != certificate.get("action_hash_a"):
            errors.append("action_a_hash_mismatch")
        if hashes.get("B") != certificate.get("action_hash_b"):
            errors.append("action_b_hash_mismatch")
    claims = contract.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("missing_claims")
        claims = []
    evidence_rows = {
        str(row.get("predicate_id")): row
        for row in certificate.get("predicates", [])
        if isinstance(row, Mapping)
    }
    covered: set[str] = set()
    for claim in claims:
        if not isinstance(claim, Mapping):
            errors.append("malformed_claim")
            continue
        predicate_id = str(claim.get("predicate_id", ""))
        row = evidence_rows.get(predicate_id)
        if row is None:
            errors.append(f"unknown_predicate:{predicate_id}")
            continue
        if claim.get("favours") != "A":
            errors.append(f"non_a_claim:{predicate_id}")
        if not isinstance(row.get("delta_a"), int) or not isinstance(row.get("delta_b"), int):
            errors.append(f"unknown_evidence:{predicate_id}")
        elif int(row["delta_a"]) <= int(row["delta_b"]):
            errors.append(f"claim_not_verified:{predicate_id}")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"missing_evidence_pointer:{predicate_id}")
        else:
            for pointer in evidence:
                if not isinstance(pointer, Mapping):
                    errors.append(f"malformed_evidence_pointer:{predicate_id}")
                    continue
                if pointer.get("source") != "observer":
                    errors.append(f"non_observer_evidence:{predicate_id}")
                if pointer.get("schema_path") not in {
                    "$.parsed",
                    "$.value_hash",
                }:
                    errors.append(f"unsupported_schema_path:{predicate_id}")
        covered.add(predicate_id)
    missing = set(certificate.get("witness_predicates", [])) - covered
    if missing:
        errors.append("uncovered_witnesses:" + ",".join(sorted(missing)))
    return {"valid": not errors, "errors": errors, "covered_witnesses": sorted(covered)}


def apply_contract_abstention(
    certificate: Mapping[str, Any], contract: Mapping[str, Any] | None
) -> dict[str, Any]:
    base_score = float(certificate.get("reverse_score", 0.5))
    if base_score != 1.0:
        return {
            "reverse_score": base_score,
            "route_to_a": False,
            "contract_applied": False,
            "contract_valid": None,
            "contract_errors": [],
        }
    if contract is None:
        return {
            "reverse_score": 0.5,
            "route_to_a": False,
            "contract_applied": True,
            "contract_valid": False,
            "contract_errors": ["missing_contract"],
        }
    validation = validate_contract(contract, certificate)
    return {
        "reverse_score": 1.0 if validation["valid"] else 0.5,
        "route_to_a": bool(validation["valid"]),
        "contract_applied": True,
        "contract_valid": bool(validation["valid"]),
        "contract_errors": validation["errors"],
    }
