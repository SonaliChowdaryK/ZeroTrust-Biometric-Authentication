# Central Zero Trust Engine Architecture

## 1. Purpose
The Central Zero Trust Engine receives evidence from four biometric modules,
validates that evidence, applies the defined decision policy, and produces
a final authentication decision with an explanation.

## 2. Biometric Modules

- M1: Face Anti-Spoofing
- M2: Face Recognition
- M3: Speaker Recognition
- M4: Voice Anti-Spoofing

## 3. Architecture Pipeline

Biometric Modules (M1-M4)
        |
        v
Evidence Input / Adapter Layer
        |
        v
Evidence Validator
        |
        v
Zero Trust Policy Evaluator
        |
        v
Final Decision Generator
        |
        v
Audit Record

## 4. Components

### 4.1 Evidence Input / Adapter Layer
Receives structured outputs from M1, M2, M3, and M4.
Each module output is converted into the common evidence format.

### 4.2 Evidence Validator
Checks:
- Required fields are present
- Module IDs are valid
- Module IDs are not duplicated
- Status values are valid
- Decision values are valid
- Evidence is structurally valid

### 4.3 Zero Trust Policy Evaluator
Applies the saved decision policy:

1. Explicit rejection from any valid module results in DENY.
2. Missing, unavailable, error, not-run, unknown, or invalid evidence
   results in REQUIRE_ADDITIONAL_VERIFICATION if no explicit rejection exists.
3. ALLOW is possible only when all four modules completed successfully
   and returned valid ACCEPT evidence.

### 4.4 Final Decision Generator
Produces one of:

- ALLOW
- DENY
- REQUIRE_ADDITIONAL_VERIFICATION

The final result includes a reason explaining the decision.

### 4.5 Audit Record
Records:
- Authentication attempt ID
- Timestamp
- Evidence received
- Validation results
- Final decision
- Reason

## 5. Security Principles

- Missing evidence must never be treated as acceptance.
- Explicit rejection takes precedence over incomplete evidence.
- Module errors must never silently result in ALLOW.
- All four modules are required for ALLOW.
- The central engine—not an individual biometric module—makes the final decision.

## 6. Limitations

- Initial prototype architecture.
- M1 and M4 integration interfaces are pending.
- The decision policy is preliminary.
- Biometric accuracy and threshold calibration require broader testing.
- This architecture does not yet represent production deployment.
