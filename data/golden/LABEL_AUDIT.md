# Golden-set label audit

The gold set remains the test set and is excluded from training and retrieval. On
2026-09-09, four policy contradictions were corrected before the new model was run:

| ID | Previous | Audited label | Rationale |
|---|---|---|---|
| GOLD-005 | routine / no escalation | delivery dispute / escalate | Repeated failed delivery while the customer was present |
| GOLD-008 | routine / no escalation | delivery dispute / escalate | Repeated delivery to unrelated neighbors despite instructions |
| GOLD-009 | routine / no escalation | delivery dispute / escalate | Package left in rain despite a designated mail hall |
| GOLD-011 | routine / no escalation | delivery dispute / escalate | Repeated delivery to another house |

No model output was consulted. `scripts/audit_gold_labels.py` applies only these
decisions and asserts the previous labels before changing them.
