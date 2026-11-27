
from typing import Any, Dict, List


VALID_MODULE_IDS = {"M1", "M2", "M3", "M4"}

VALID_STATUSES = {
    "completed",
    "error",
    "unavailable",
    "not_run"
}

VALID_DECISIONS = {
    "ACCEPT",
    "REJECT",
    "UNKNOWN"
}

REQUIRED_FIELDS = {
    "module_id",
    "module_name",
    "status",
    "decision",
    "verified"
}


def validate_module_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:

    errors = []

    if not isinstance(evidence, dict):
        return {
            "valid": False,
            "module_id": None,
            "errors": ["Evidence must be a dictionary."]
        }

    missing_fields = REQUIRED_FIELDS - set(evidence.keys())

    if missing_fields:
        return {
            "valid": False,
            "module_id": evidence.get("module_id"),
            "errors:": [],
            "errors": [
                f"Missing required fields: {sorted(missing_fields)}"
            ]
        }

    module_id = evidence["module_id"]
    module_name = evidence["module_name"]
    status = evidence["status"]
    decision = evidence["decision"]
    verified = evidence["verified"]

    # Module ID: check type before set membership
    if not isinstance(module_id, str):
        errors.append("module_id must be a string.")
    elif module_id not in VALID_MODULE_IDS:
        errors.append(f"Invalid module_id: {module_id}")

    # Module name
    if not isinstance(module_name, str) or not module_name.strip():
        errors.append("module_name must be a non-empty string.")

    # Status: check type before set membership
    if not isinstance(status, str):
        errors.append("status must be a string.")
    elif status not in VALID_STATUSES:
        errors.append(f"Invalid status: {status}")

    # Decision: check type before set membership
    if not isinstance(decision, str):
        errors.append("decision must be a string.")
    elif decision not in VALID_DECISIONS:
        errors.append(f"Invalid decision: {decision}")

    # verified must be Boolean or None
    if verified is not None and type(verified) is not bool:
        errors.append("verified must be True, False, or None.")

    # Consistency checks only when decision is a valid string
    if decision == "ACCEPT" and verified is not True:
        errors.append("ACCEPT decision requires verified=True.")

    if decision == "REJECT" and verified is not False:
        errors.append("REJECT decision requires verified=False.")

    if decision == "UNKNOWN" and verified is not None:
        errors.append("UNKNOWN decision requires verified=None.")

    # Failed modules cannot claim successful verification
    if isinstance(status, str) and status in {
        "error", "unavailable", "not_run"
    }:
        if decision == "ACCEPT" or verified is True:
            errors.append(
                f"Status '{status}' cannot claim successful verification."
            )

    return {
        "valid": len(errors) == 0,
        "module_id": module_id,
        "errors": errors
    }


def validate_all_module_evidence(
    evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not isinstance(evidence_list, list):
        return {
            "valid": False,
            "module_results": [],
            "errors": ["Evidence input must be a list."]
        }

    module_results = []
    global_errors = []
    seen_module_ids = set()

    for evidence in evidence_list:

        result = validate_module_evidence(evidence)
        module_results.append(result)

        module_id = result.get("module_id")

        # Only string IDs can safely be checked for duplicates
        if isinstance(module_id, str) and module_id in VALID_MODULE_IDS:

            if module_id in seen_module_ids:
                global_errors.append(
                    f"Duplicate evidence received for module {module_id}."
                )
            else:
                seen_module_ids.add(module_id)

    all_modules_valid = all(
        result["valid"] for result in module_results
    )

    overall_valid = (
        all_modules_valid
        and len(global_errors) == 0
    )

    return {
        "valid": overall_valid,
        "module_results": module_results,
        "errors": global_errors
    }
