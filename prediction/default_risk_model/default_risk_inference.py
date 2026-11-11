"""Standalone CPU inference for the LendingClub 36-month default-risk model.

Needs: numpy (+ lightgbm only when model_type == "lightgbm"). No torch, pandas, Modal or training data.
Scope: US unsecured 36-month LendingClub loans, 2009-2015 vintages. Prototype; not evidence about any other lender.
Required inputs: loan_amnt, annual_inc, dti, fico_score (or fico_range_low + fico_range_high); training excluded loans
lacking them. Everything else is optional and handled as in training. Risk_category is a HEURISTIC band, not a validated class.

    from default_risk_inference import predict_default_risk
    predict_default_risk({"loan_amnt": 12000, "annual_inc": 65000, "dti": 18.5, "fico_score": 690,
                          "home_ownership": "RENT", "purpose": "debt_consolidation", "emp_length": "5 years",
                          "revol_util": 54.0, "revol_bal": 11000, "open_acc": 9, "total_acc": 21,
                          "credit_history_months": 150})
"""
import json
import math
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_CACHE = {}
_EMP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
        "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
_REQUIRED = ("loan_amnt", "annual_inc", "dti", "fico_score")
LABELS = {0: "Non-default", 1: "Default"}
RISK_BAND_NOTE = ("Heuristic bands, not independently validated: High = calibrated p >= decision threshold; "
                  "Low = p below a fixed fraction of that threshold (see model_info.json); Medium = in between.")


def _to_float(v):
    if v is None:
        return float("nan")
    if isinstance(v, str):
        s = v.strip().rstrip("%").replace(",", "")
        if not s:
            return float("nan")
        try:
            x = float(s)
        except ValueError:
            return float("nan")
    else:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
    return x if math.isfinite(x) else float("nan")


def _emp_length(v):
    if isinstance(v, str) and v.strip() in _EMP:
        return float(_EMP[v.strip()])
    return _to_float(v)


def _norm_cat(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    return s or None


def _numeric_values(applicant, spec):
    vals, warns = [], []
    for c in spec["numeric"]:
        raw = applicant.get(c)
        x = _emp_length(raw) if c == "emp_length" else _to_float(raw)
        if not math.isnan(x):
            lo, hi = spec["clip"][c]
            if x < lo or x > hi:
                warns.append("%s=%s outside the typical training range [%.4g, %.4g]; clipped" % (c, raw, lo, hi))
                x = min(max(x, lo), hi)
        vals.append(x)
    return vals, warns


def _cat_codes(applicant, spec):
    codes = []
    for c in spec["categorical"]:
        cats = spec["categories"][c]
        s = _norm_cat(applicant.get(c))
        codes.append(cats.index(s) if s in cats else cats.index("__other__"))
    return codes


def build_lgb_row(applicant, spec):
    nums, warns = _numeric_values(applicant, spec)
    return np.asarray([nums + [float(k) for k in _cat_codes(applicant, spec)]], dtype=np.float64), warns


def build_mlp_row(applicant, spec):
    nums, warns = _numeric_values(applicant, spec)
    m = spec["mlp"]
    out, isna = [], {}
    for c, x in zip(spec["numeric"], nums):
        isna[c] = math.isnan(x)
        if not isna[c] and c in spec["log1p"]:
            x = math.log1p(max(x, 0.0))
        if isna[c]:
            x = m["median"][c]
        out.append((x - m["mean"][c]) / m["std"][c])
    out.extend(1.0 if isna[c] else 0.0 for c in m["indicator_cols"])
    for c, k in zip(spec["categorical"], _cat_codes(applicant, spec)):
        oh = [0.0] * len(spec["categories"][c])
        oh[k] = 1.0
        out.extend(oh)
    return np.asarray([out], dtype=np.float64), warns


def _mlp_forward(x, w):
    h = np.asarray(x, dtype=np.float64)
    for i in range(int(w["n_hidden"])):
        h = h @ w["W%d" % i].T + w["b%d" % i]
        h = (h - w["rm%d" % i]) / np.sqrt(w["rv%d" % i] + float(w["eps"])) * w["g%d" % i] + w["be%d" % i]
        h = np.maximum(h, 0.0)
    z = h @ w["Wout"].T + w["bout"]
    return 1.0 / (1.0 + np.exp(-z.ravel()))


def apply_calibration(p, cal):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    m = cal["method"]
    if m == "none":
        return p
    if m == "platt":
        return 1.0 / (1.0 + np.exp(-(cal["a"] * np.log(p / (1.0 - p)) + cal["b"])))
    if m == "isotonic":
        return np.interp(p, cal["x"], cal["y"])
    raise ValueError("unknown calibration method %r" % m)


def risk_category(p, threshold, low_max):
    if p >= threshold:
        return "High"
    if p < low_max:
        return "Low"
    return "Medium"


def _imputed_optional(a, spec):
    """Optional inputs that were missing and will be handled exactly as in training (not an error)."""
    out = []
    for c in spec["numeric"]:
        if c in spec["required"]:
            continue
        x = _emp_length(a.get(c)) if c == "emp_length" else _to_float(a.get(c))
        if math.isnan(x):
            out.append(c)
    out.extend(c for c in spec["categorical"] if c not in spec["required"] and _norm_cat(a.get(c)) is None)
    return out


def _load(artifact_dir=None):
    d = Path(artifact_dir) if artifact_dir else _DIR
    key = str(d.resolve())
    if key not in _CACHE:
        info = json.loads((d / "model_info.json").read_text())
        spec = json.loads((d / "preprocessing_spec.json").read_text())
        if info["model_type"] == "lightgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(d / info["model_file"]))
        else:
            w = np.load(d / info["model_file"])
            model = {k: w[k] for k in w.files}
        _CACHE[key] = {"info": info, "spec": spec, "model": model}
    return _CACHE[key]


def raw_probability(applicant, artifact_dir=None):
    art = _load(artifact_dir)
    if art["info"]["model_type"] == "lightgbm":
        x, warns = build_lgb_row(applicant, art["spec"])
        return float(art["model"].predict(x)[0]), warns
    x, warns = build_mlp_row(applicant, art["spec"])
    return float(_mlp_forward(x, art["model"])[0]), warns


def predict_default_risk(applicant_data, artifact_dir=None):
    """Dict in -> dict out (a list of dicts returns a list). Missing optional fields are handled exactly as in training."""
    if isinstance(applicant_data, (list, tuple)):
        return [predict_default_risk(a, artifact_dir) for a in applicant_data]
    a = dict(applicant_data)
    if math.isnan(_to_float(a.get("fico_score"))) and "fico_range_low" in a and "fico_range_high" in a:
        a["fico_score"] = (_to_float(a["fico_range_low"]) + _to_float(a["fico_range_high"])) / 2.0
    art = _load(artifact_dir)
    info = art["info"]
    required = art["spec"].get("required", list(_REQUIRED))     # same list the training cohort enforced
    missing = [c for c in required if math.isnan(_to_float(a.get(c)))]
    if missing:
        raise ValueError("missing or non-numeric required fields: %s" % missing)
    raw, warns = raw_probability(a, artifact_dir)
    p = float(apply_calibration(raw, info["calibration"]))
    thr = float(info["decision_threshold"])
    cls = int(p >= thr)
    return {"default_probability": p, "predicted_class": cls, "predicted_label": LABELS[cls],
            "risk_category": risk_category(p, thr, float(info["risk_low_max"])),
            "risk_category_note": info.get("risk_bands_note", RISK_BAND_NOTE), "decision_threshold": thr,
            "imputed_optional_fields": _imputed_optional(a, art["spec"]), "warnings": warns}


if __name__ == "__main__":
    print(predict_default_risk({"loan_amnt": 12000, "annual_inc": 65000, "dti": 18.5, "fico_score": 690,
                                "home_ownership": "RENT", "purpose": "debt_consolidation", "emp_length": "5 years",
                                "revol_util": 54.0, "revol_bal": 11000, "open_acc": 9, "total_acc": 21,
                                "credit_history_months": 150}))
