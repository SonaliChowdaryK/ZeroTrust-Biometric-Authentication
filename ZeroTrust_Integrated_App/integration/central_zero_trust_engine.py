
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------
# 1. Ensure integration directory is importable
# --------------------------------------------------

INTEGRATION_DIR = Path(__file__).resolve().parent

if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))


# --------------------------------------------------
# 2. Import frozen components
# --------------------------------------------------

from zero_trust_policy_evaluator import evaluate_zero_trust_policy
from decision_output import generate_decision_output


# --------------------------------------------------
# 3. Central Zero Trust Engine
# --------------------------------------------------

def run_zero_trust_engine(
    module_evidence: Any
) -> dict:
    """
    Central orchestration layer.

    Pipeline:
        Module Evidence
            -> Policy Evaluator
            -> Decision Output Formatter
            -> Final Standardized Output

    The orchestrator does not independently make
    authentication decisions.
    """

    # --------------------------------------------------
    # A. Safely handle invalid top-level input
    # --------------------------------------------------

    if not isinstance(module_evidence, list):

        policy_result = evaluate_zero_trust_policy(
            module_evidence
        )

        return generate_decision_output(policy_result)

    # --------------------------------------------------
    # B. Run Zero Trust Policy Evaluation
    # --------------------------------------------------

    policy_result = evaluate_zero_trust_policy(
        module_evidence
    )

    # --------------------------------------------------
    # C. Format standardized output and audit record
    # --------------------------------------------------

    final_output = generate_decision_output(
        policy_result
    )

    return final_output
