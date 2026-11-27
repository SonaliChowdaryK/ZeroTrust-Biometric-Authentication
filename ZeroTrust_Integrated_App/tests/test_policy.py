import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from central_zero_trust_engine import run_zero_trust_engine

def ev(mid, decision="ACCEPT", status="completed"):
    return {"module_id":mid, "module_name":mid, "status":status,
            "decision":decision,
            "verified": True if decision=="ACCEPT" else False if decision=="REJECT" else None}

def all_accept():
    return [ev("M1"),ev("M2"),ev("M3"),ev("M4")]

def test_all_accept_allow():
    assert run_zero_trust_engine(all_accept())["final_decision"] == "ALLOW"

def test_any_valid_reject_denies():
    items=[ev("M1"),ev("M2","REJECT"),ev("M3"),ev("M4")]
    assert run_zero_trust_engine(items)["final_decision"] == "DENY"

def test_missing_module_requires_additional():
    assert run_zero_trust_engine(all_accept()[:3])["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"

def test_error_requires_additional():
    items=all_accept(); items[1]=ev("M2","UNKNOWN","error")
    assert run_zero_trust_engine(items)["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"

def test_duplicate_requires_additional():
    items=all_accept()+[ev("M1")]
    assert run_zero_trust_engine(items)["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"

def test_invalid_evidence_requires_additional():
    items=all_accept(); items[0]["verified"]="yes"
    assert run_zero_trust_engine(items)["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"

def test_reject_precedes_missing():
    items=[ev("M3","REJECT")]
    assert run_zero_trust_engine(items)["final_decision"] == "DENY"

def test_not_run_requires_additional():
    items=all_accept(); items[3]=ev("M4","UNKNOWN","not_run")
    assert run_zero_trust_engine(items)["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"

def test_non_list_requires_additional():
    assert run_zero_trust_engine(None)["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"
