
from typing import Any, Dict, List

from evidence_validator import validate_all_module_evidence


REQUIRED_MODULES = {"M1", "M2", "M3", "M4"}


def evaluate_zero_trust_policy(
    evidence_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Central Zero Trust Policy Evaluator.

    Possible decisions:
        ALLOW
        DENY
        REQUIRE_ADDITIONAL_VERIFICATION

    Security rules:
        1. A valid completed REJECT from a module -> DENY.
        2. Invalid evidence -> additional verification.
        3. Missing/incomplete modules -> additional verification.
        4. ALLOW only if all four required modules completed
           and returned valid ACCEPT evidence.
    """

    # --------------------------------------------------
    # 0. SAFETY CHECK: Validate top-level input type
    # --------------------------------------------------

    if not isinstance(evidence_list, list):

        return {
            "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
            "reason": "Evidence input must be a list.",
            "validation": {
                "valid": False,
                "module_results": [],
                "errors": [
                    "Evidence input must be a list."
                ]
            },
            "module_evidence": []
        }

    # --------------------------------------------------
    # 1. VALIDATE ALL MODULE EVIDENCE
    # --------------------------------------------------

    validation_result = validate_all_module_evidence(
        evidence_list
    )

    module_results = validation_result["module_results"]

    # --------------------------------------------------
    # 2. BUILD MODULE EVIDENCE LOOKUP
    # --------------------------------------------------

    evidence_by_module = {}

    for evidence in evidence_list:

        if isinstance(evidence, dict):

            module_id = evidence.get("module_id")

            if isinstance(module_id, str):

                evidence_by_module.setdefault(
                    module_id, []
                ).append(evidence)

    # --------------------------------------------------
    # 3. CHECK FOR VALID COMPLETED REJECTION
    # --------------------------------------------------
    # A rejection is trusted only if:
    #   - evidence is structurally valid
    #   - module status is completed
    #   - decision is REJECT
    #   - verified is False

    valid_rejection_found = False

    for evidence, validation in zip(
        evidence_list,
        module_results
    ):

        if not isinstance(evidence, dict):
            continue

        if not validation["valid"]:
            continue

        if (
            evidence.get("status") == "completed"
            and evidence.get("decision") == "REJECT"
            and evidence.get("verified") is False
        ):

            valid_rejection_found = True
            break

    # Valid rejection takes precedence
    if valid_rejection_found:

        return {
            "final_decision": "DENY",
            "reason": (
                "A required biometric module produced valid, "
                "completed rejection evidence."
            ),
            "validation": validation_result,
            "module_evidence": evidence_list
        }

    # --------------------------------------------------
    # 4. INVALID EVIDENCE -> ADDITIONAL VERIFICATION
    # --------------------------------------------------

    if not validation_result["valid"]:

        return {
            "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
            "reason": (
                "Evidence validation failed. The received evidence "
                "cannot safely support an ALLOW decision."
            ),
            "validation": validation_result,
            "module_evidence": evidence_list
        }

    # --------------------------------------------------
    # 5. CHECK ALL REQUIRED MODULES ARE PRESENT EXACTLY ONCE
    # --------------------------------------------------

    for module_id in REQUIRED_MODULES:

        # Missing module
        if module_id not in evidence_by_module:

            return {
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": (
                    f"Required module {module_id} is missing."
                ),
                "validation": validation_result,
                "module_evidence": evidence_list
            }

        # Duplicate module
        if len(evidence_by_module[module_id]) != 1:

            return {
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": (
                    f"Duplicate evidence received for {module_id}."
                ),
                "validation": validation_result,
                "module_evidence": evidence_list
            }

    # --------------------------------------------------
    # 6. CHECK EACH REQUIRED MODULE COMPLETED AND ACCEPTED
    # --------------------------------------------------

    for module_id in REQUIRED_MODULES:

        evidence = evidence_by_module[module_id][0]

        # Module must complete successfully
        if evidence["status"] != "completed":

            return {
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": (
                    f"Required module {module_id} "
                    "did not complete successfully."
                ),
                "validation": validation_result,
                "module_evidence": evidence_list
            }

        # Module must explicitly ACCEPT
        if evidence["decision"] != "ACCEPT":

            return {
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": (
                    f"Required module {module_id} returned "
                    "UNKNOWN or a non-acceptance result."
                ),
                "validation": validation_result,
                "module_evidence": evidence_list
            }

    # --------------------------------------------------
    # 7. ALLOW ONLY WHEN ALL FOUR MODULES ACCEPT
    # --------------------------------------------------

    return {
        "final_decision": "ALLOW",
        "reason": (
            "All four required biometric modules completed "
            "successfully and returned valid ACCEPT evidence."
        ),
        "validation": validation_result,
        "module_evidence": evidence_list
    }
