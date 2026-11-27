
import uuid
from datetime import datetime, timezone
from typing import Any, Dict


VALID_FINAL_DECISIONS = {
    "ALLOW",
    "DENY",
    "REQUIRE_ADDITIONAL_VERIFICATION"
}

REQUIRED_MODULES = {"M1", "M2", "M3", "M4"}


def generate_decision_output(
    policy_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standardize the Central Zero Trust Engine output.

    This component does not independently authenticate a user.
    It formats the Policy Evaluator result and safely handles
    malformed or inconsistent inputs.
    """

    attempt_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------
    # 1. Validate top-level input
    # --------------------------------------------------

    if not isinstance(policy_result, dict):

        return {
            "attempt_id": attempt_id,
            "timestamp": timestamp,
            "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
            "reason": "Invalid policy result received.",
            "validation_passed": False,
            "module_summary": [],
            "audit_record": {
                "attempt_id": attempt_id,
                "timestamp": timestamp,
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": "Invalid policy result received.",
                "validation_passed": False,
                "module_summary": [],
                "record_status": "invalid_input"
            }
        }

    # --------------------------------------------------
    # 2. Validate decision type and value
    # --------------------------------------------------

    final_decision = policy_result.get("final_decision")
    reason = policy_result.get("reason")

    if not isinstance(final_decision, str):

        final_decision = "REQUIRE_ADDITIONAL_VERIFICATION"
        reason = "Invalid decision type received."

    elif final_decision not in VALID_FINAL_DECISIONS:

        final_decision = "REQUIRE_ADDITIONAL_VERIFICATION"
        reason = "Invalid or missing final decision from policy evaluator."

    # --------------------------------------------------
    # 3. Validate validation information
    # --------------------------------------------------

    validation = policy_result.get("validation")

    validation_passed = (
        isinstance(validation, dict)
        and validation.get("valid") is True
    )

    # --------------------------------------------------
    # 4. Validate module evidence container
    # --------------------------------------------------

    module_evidence = policy_result.get("module_evidence")

    if not isinstance(module_evidence, list):

        module_evidence = []

        if final_decision == "ALLOW":

            final_decision = "REQUIRE_ADDITIONAL_VERIFICATION"
            reason = "Malformed module evidence cannot support ALLOW."

    # --------------------------------------------------
    # 5. Create safe module summary
    # --------------------------------------------------

    safe_module_summary = []
    malformed_module_evidence = False

    for module in module_evidence:

        if not isinstance(module, dict):

            malformed_module_evidence = True
            continue

        module_id = module.get("module_id")
        module_name = module.get("module_name")
        status = module.get("status")
        decision = module.get("decision")
        verified = module.get("verified")

        # Validate basic types
        if (
            not isinstance(module_id, str)
            or not isinstance(module_name, str)
            or not isinstance(status, str)
            or not isinstance(decision, str)
            or (verified is not None and type(verified) is not bool)
        ):

            malformed_module_evidence = True
            continue

        safe_module_summary.append({
            "module_id": module_id,
            "module_name": module_name,
            "status": status,
            "decision": decision,
            "verified": verified
        })

    # --------------------------------------------------
    # 6. Additional consistency check for ALLOW
    # --------------------------------------------------

    if final_decision == "ALLOW":

        summary_module_ids = [
            module["module_id"]
            for module in safe_module_summary
        ]

        all_modules_present = (
            set(summary_module_ids) == REQUIRED_MODULES
            and len(summary_module_ids) == 4
        )

        all_modules_accept = all(
            module["status"] == "completed"
            and module["decision"] == "ACCEPT"
            and module["verified"] is True
            for module in safe_module_summary
        )

        if (
            not validation_passed
            or malformed_module_evidence
            or not all_modules_present
            or not all_modules_accept
        ):

            final_decision = "REQUIRE_ADDITIONAL_VERIFICATION"
            reason = (
                "Output consistency checks failed. "
                "The evidence cannot safely support ALLOW."
            )

    # --------------------------------------------------
    # 7. Ensure reason is a safe string
    # --------------------------------------------------

    if not isinstance(reason, str) or not reason.strip():

        reason = "No valid decision reason was provided."

    # --------------------------------------------------
    # 8. Create standardized output
    # --------------------------------------------------

    output = {
        "attempt_id": attempt_id,
        "timestamp": timestamp,
        "final_decision": final_decision,
        "reason": reason,
        "validation_passed": validation_passed,
        "module_summary": safe_module_summary
    }

    # --------------------------------------------------
    # 9. Create privacy-conscious audit record
    # --------------------------------------------------

    audit_record = {
        "attempt_id": attempt_id,
        "timestamp": timestamp,
        "final_decision": final_decision,
        "reason": reason,
        "validation_passed": validation_passed,
        "module_summary": safe_module_summary,
        "record_status": "completed"
    }

    return {
        **output,
        "audit_record": audit_record
    }
