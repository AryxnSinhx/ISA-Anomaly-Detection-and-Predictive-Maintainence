def decision_logic(
    anomaly_score: float,
    rul: float,
    critical_anomaly_thresh: float = -0.3,
    warning_anomaly_thresh:  float = -0.2,
    critical_rul_thresh:     float = 20,
    warning_rul_thresh:      float = 15,
) -> str:
    """
    Maps (anomaly_score, RUL) to an actionable maintenance decision.

    Parameters
    ----------
    anomaly_score            : IsolationForest decision_function output.
                               More negative = more anomalous.
    rul                      : Predicted remaining useful life (cycles).
    critical_anomaly_thresh  : Score below this triggers critical path.
                               Default -0.3 — tune per dataset on val set.
    warning_anomaly_thresh   : Score below this triggers backup path.
                               Default -0.2 — tune per dataset on val set.
    critical_rul_thresh      : RUL below this (combined with anomaly) → shutdown.
                               Default 20 cycles.
    warning_rul_thresh       : RUL below this alone → alert ground control.
                               Default 15 cycles.

    Returns
    -------
    str : One of four decision strings.
    """

    if anomaly_score < critical_anomaly_thresh and rul < critical_rul_thresh:
        return "CRITICAL FAILURE — SHUTDOWN"

    elif anomaly_score < warning_anomaly_thresh:
        return "Switch to backup subsystem"

    elif rul < warning_rul_thresh:
        return "Alert ground control"

    else:
        return "System healthy"
