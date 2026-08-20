 #!/usr/bin/env python3
"""
Statistical analysis workflow for the Second Victim Phenomenon survey.

The script does not modify the source CSV. It reads the Google Forms export,
cleans analysis copies of the data, scores survey scales, runs descriptive
statistics, association tests, and ordinary least squares regression models,
then writes reusable outputs to analysis_outputs/.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from textwrap import dedent


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "openpyxl": "openpyxl",
}


def import_or_exit():
    missing = []
    modules = {}
    for import_name in REQUIRED_PACKAGES:
        try:
            modules[import_name] = __import__(import_name)
        except ModuleNotFoundError:
            missing.append(REQUIRED_PACKAGES[import_name])

    if missing:
        pkg_list = " ".join(sorted(set(missing)))
        print(
            dedent(
                f"""
                Missing Python packages: {pkg_list}

                Run these commands from this folder:
                  python3 -m venv .venv
                  source .venv/bin/activate
                  python -m pip install --upgrade pip
                  python -m pip install {pkg_list}
                  python analysis.py
                """
            ).strip(),
            file=sys.stderr,
        )
        sys.exit(1)

    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    return np, pd, stats, sm, het_breuschpagan, multipletests, variance_inflation_factor


np, pd, stats, sm, het_breuschpagan, multipletests, variance_inflation_factor = import_or_exit()


DEFAULT_INPUT = "Second victim phenomenon  - Form responses 1.csv"
DEFAULT_OUTPUT_DIR = "analysis_outputs"

PHQ_ITEMS = list(range(5, 14))
GAD_ITEMS = list(range(14, 21))
SV_EMOTIONAL_ITEMS = list(range(21, 30))
PEER_SUPPORT_ITEMS = list(range(30, 34))
SUPERVISOR_SUPPORT_ITEMS = list(range(34, 38))
ORG_SUPPORT_ITEMS = list(range(38, 41))
PERFORMANCE_ITEMS = list(range(41, 45))
INTENT_LEAVE_ITEMS = list(range(45, 49))
WORK_WITHDRAWAL_ITEMS = list(range(49, 52))
PROFESSIONAL_GROWTH_ITEMS = list(range(52, 56))
RESOURCE_NEED_ITEMS = list(range(56, 63))
BURNOUT_ITEMS = list(range(63, 79))

REVERSE_LIKERT_ITEM_INDICES = {
    # Peer/supervisor/organisation support: reverse negatively worded items so
    # higher scores mean more support.
    30,
    32,
    33,
    36,
    40,
    # Burnout: reverse positively worded OLBI-style items so higher scores mean
    # higher burnout.
    63,
    67,
    69,
    72,
    74,
    76,
    77,
    78,
}

SCALE_DEFINITIONS = [
    ("PHQ9_Total", "PHQ-9 depression total", PHQ_ITEMS, "phq", False),
    ("GAD7_Total", "GAD-7 anxiety total", GAD_ITEMS, "phq", False),
    ("SV_Emotional_Total", "Second victim emotional impact", SV_EMOTIONAL_ITEMS, "likert", False),
    ("Peer_Support_Total", "Peer support", PEER_SUPPORT_ITEMS, "likert", True),
    ("Supervisor_Support_Total", "Supervisor support", SUPERVISOR_SUPPORT_ITEMS, "likert", True),
    ("Org_Support_Total", "Organisation support", ORG_SUPPORT_ITEMS, "likert", True),
    ("Performance_Impact_Total", "Performance impact", PERFORMANCE_ITEMS, "likert", False),
    ("Intent_Leave_Total", "Intention to leave", INTENT_LEAVE_ITEMS, "likert", False),
    ("Work_Withdrawal_Total", "Mental health day/time off/distraction impact", WORK_WITHDRAWAL_ITEMS, "likert", False),
    ("Professional_Growth_Total", "Professional growth", PROFESSIONAL_GROWTH_ITEMS, "likert", False),
    ("Resource_Need_Total", "Resource/support need", RESOURCE_NEED_ITEMS, "desired", False),
    ("Burnout_Total", "Burnout total", BURNOUT_ITEMS, "likert", True),
]

PRIMARY_CONTINUOUS = [
    "Age",
    "PHQ9_Total",
    "GAD7_Total",
    "SV_Emotional_Total",
    "Peer_Support_Total",
    "Supervisor_Support_Total",
    "Org_Support_Total",
    "Performance_Impact_Total",
    "Intent_Leave_Total",
    "Work_Withdrawal_Total",
    "Professional_Growth_Total",
    "Resource_Need_Total",
    "Burnout_Total",
]

SCALE_INDEX_BY_NAME = {name: indices for name, _, indices, _, _ in SCALE_DEFINITIONS}

DISPLAY_LABELS = {
    "Age": "age",
    "PHQ9_Total": "PHQ-9 depression score",
    "GAD7_Total": "GAD-7 anxiety score",
    "SV_Emotional_Total": "second victim emotional impact",
    "Peer_Support_Total": "peer support",
    "Supervisor_Support_Total": "supervisor support",
    "Org_Support_Total": "organisational support",
    "Performance_Impact_Total": "performance impact",
    "Intent_Leave_Total": "intention to leave",
    "Work_Withdrawal_Total": "work withdrawal / mental-health-day impact",
    "Professional_Growth_Total": "professional growth",
    "Resource_Need_Total": "support resource need",
    "Burnout_Total": "burnout",
    "PHQ9_Category": "PHQ-9 category",
    "GAD7_Category": "GAD-7 category",
    "PHQ9_ModeratePlus": "moderate-or-higher PHQ-9 burden",
    "GAD7_ModeratePlus": "moderate-or-higher GAD-7 burden",
    "Gender": "gender",
    "Designation": "designation",
    "Specialization": "specialization",
}

RELATIONSHIP_OUTCOMES = [
    "PHQ9_Total",
    "GAD7_Total",
    "Burnout_Total",
    "Intent_Leave_Total",
    "Performance_Impact_Total",
    "Work_Withdrawal_Total",
]


def norm_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else np.nan


def make_unique_headers(headers):
    seen = {}
    cleaned = []
    for header in headers:
        base = norm_text(header) or "Unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1
        cleaned.append(base if count == 0 else f"{base}.{count + 1}")
    return cleaned


def inspect_raw_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row_lengths = []
        for i, row in enumerate(reader, start=2):
            row_lengths.append((i, len(row)))

    duplicate_raw = sorted({h for h in headers if headers.count(h) > 1})
    trimmed = [norm_text(h) or "" for h in headers]
    duplicate_trimmed = sorted({h for h in trimmed if trimmed.count(h) > 1})
    blank_headers = [i + 1 for i, h in enumerate(headers) if not str(h).strip()]
    headers_with_edge_spaces = [h for h in headers if h != str(h).strip()]
    bad_row_lengths = [
        {"Row": row_num, "Columns found": length, "Expected columns": len(headers)}
        for row_num, length in row_lengths
        if length != len(headers)
    ]
    return {
        "raw_headers": headers,
        "trimmed_headers": trimmed,
        "duplicate_raw": duplicate_raw,
        "duplicate_trimmed": duplicate_trimmed,
        "blank_headers": blank_headers,
        "headers_with_edge_spaces": headers_with_edge_spaces,
        "bad_row_lengths": bad_row_lengths,
        "n_data_rows": len(row_lengths),
        "n_columns": len(headers),
    }


def load_data(path, raw_info):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = make_unique_headers(list(df.columns))
    for col in df.columns:
        df[col] = df[col].map(norm_text)
    return df


def value_key(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def score_phq_value(value):
    mapping = {
        "not at all": 0,
        "several days": 1,
        "more than half the days": 2,
        "over the half days": 2,
        "over half the days": 2,
        "nearly everyday": 3,
        "nearly every day": 3,
    }
    return mapping.get(value_key(value), np.nan)


def score_likert_value(value, reverse=False):
    mapping = {
        "strongly disagree": 1,
        "disagree": 2,
        "neutral": 3,
        "agree": 4,
        "strongly agree": 5,
    }
    score = mapping.get(value_key(value), np.nan)
    if pd.isna(score):
        return np.nan
    return 6 - score if reverse else score


def score_desired_value(value):
    mapping = {
        "strongly not desired": 1,
        "not desired": 2,
        "undesired": 2,
        "neutral": 3,
        "desired": 4,
        "strongly desired": 5,
    }
    return mapping.get(value_key(value), np.nan)


def safe_cols(df, indices):
    return [df.columns[i] for i in indices if i < len(df.columns)]


def add_scores(df, raw_info):
    scored = df.copy()
    scored["Age"] = pd.to_numeric(scored["Age"], errors="coerce")
    scored["Timestamp"] = pd.to_datetime(scored["Timestamp"], errors="coerce", dayfirst=True)

    item_codebook = []
    for i, col in enumerate(scored.columns):
        if i in PHQ_ITEMS:
            score_col = f"item_{i:02d}_score"
            scored[score_col] = scored[col].map(score_phq_value)
            item_codebook.append((score_col, col, "PHQ/GAD frequency", "0-3"))
        elif i in GAD_ITEMS:
            score_col = f"item_{i:02d}_score"
            scored[score_col] = scored[col].map(score_phq_value)
            item_codebook.append((score_col, col, "PHQ/GAD frequency", "0-3"))
        elif i in range(21, 79):
            reverse = i in REVERSE_LIKERT_ITEM_INDICES
            score_col = f"item_{i:02d}_score"
            if i in RESOURCE_NEED_ITEMS:
                scored[score_col] = scored[col].map(score_desired_value)
                scale_type = "Resource need"
            else:
                scored[score_col] = scored[col].map(lambda x, rev=reverse: score_likert_value(x, reverse=rev))
                scale_type = "Likert reverse-coded" if reverse else "Likert"
            item_codebook.append((score_col, col, scale_type, "1-5"))

    scale_metadata = []
    for score_name, label, indices, _, uses_reverse in SCALE_DEFINITIONS:
        item_score_cols = [f"item_{i:02d}_score" for i in indices if f"item_{i:02d}_score" in scored.columns]
        scored[score_name] = scored[item_score_cols].mean(axis=1, skipna=True)
        if score_name in ("PHQ9_Total", "GAD7_Total"):
            scored[score_name] = scored[item_score_cols].sum(axis=1, min_count=len(item_score_cols))
        scale_metadata.append(
            {
                "Scale": score_name,
                "Label": label,
                "Items": len(item_score_cols),
                "Aggregation": "sum" if score_name in ("PHQ9_Total", "GAD7_Total") else "mean",
                "Reverse coded items included": "Yes" if uses_reverse else "No",
                "Item score columns": ", ".join(item_score_cols),
            }
        )

    scored["PHQ9_Category"] = pd.cut(
        scored["PHQ9_Total"],
        bins=[-0.01, 4, 9, 14, 19, 27],
        labels=["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"],
    )
    scored["PHQ9_ModeratePlus"] = np.where(scored["PHQ9_Total"] >= 10, "Yes", "No")
    scored.loc[scored["PHQ9_Total"].isna(), "PHQ9_ModeratePlus"] = np.nan

    scored["GAD7_Category"] = pd.cut(
        scored["GAD7_Total"],
        bins=[-0.01, 4, 9, 14, 21],
        labels=["Minimal", "Mild", "Moderate", "Severe"],
    )
    scored["GAD7_ModeratePlus"] = np.where(scored["GAD7_Total"] >= 10, "Yes", "No")
    scored.loc[scored["GAD7_Total"].isna(), "GAD7_ModeratePlus"] = np.nan

    return scored, pd.DataFrame(item_codebook, columns=["Score column", "Original question", "Scoring", "Range"]), pd.DataFrame(scale_metadata)


def apply_analysis_fixes(scored):
    """Create an analysis-ready copy with transparent fixes for known issues."""
    fixed = scored.copy()
    fix_rows = [
        {
            "Issue": "Headers contain leading/trailing spaces",
            "Fix applied in workbook": "All output sheets use trimmed headers with repeated whitespace collapsed.",
            "Changes made": "Source CSV left unchanged.",
            "Interpretation note": "Use Scored_Data_AnalysisReady for statistical work.",
        }
    ]

    if "Specialization" in fixed.columns:
        missing = fixed["Specialization"].isna()
        fixed["Specialization_Missing_Flag"] = missing.map({True: "Missing in source", False: ""})
        fixed.loc[missing, "Specialization"] = "Not specified"
        fix_rows.append(
            {
                "Issue": "Missing Specialization values",
                "Fix applied in workbook": "Blank Specialization values set to 'Not specified' in analysis-ready data.",
                "Changes made": f"{int(missing.sum())} records flagged in Specialization_Missing_Flag.",
                "Interpretation note": "These records remain usable in analyses; treat 'Not specified' as its own category.",
            }
        )

    imputation_rows = []
    for score_name, label, indices, _, _ in SCALE_DEFINITIONS:
        item_cols = [f"item_{i:02d}_score" for i in indices if f"item_{i:02d}_score" in fixed.columns]
        if not item_cols:
            continue

        original_col = f"{score_name}_Original"
        fixed[original_col] = fixed[score_name]
        missing_count = fixed[item_cols].isna().sum(axis=1)
        answered_count = fixed[item_cols].notna().sum(axis=1)
        threshold = max(1, int(np.floor(len(item_cols) * 0.2)))
        eligible = (missing_count > 0) & (missing_count <= threshold) & (answered_count > 0)
        scale_median_impute = missing_count > threshold
        row_mean = fixed[item_cols].mean(axis=1, skipna=True)
        imputed_items = fixed[item_cols].T.fillna(row_mean).T

        if score_name in ("PHQ9_Total", "GAD7_Total"):
            fixed_score = imputed_items.sum(axis=1)
        else:
            fixed_score = imputed_items.mean(axis=1)

        fixed.loc[eligible, score_name] = fixed_score.loc[eligible]
        scale_median = fixed.loc[~scale_median_impute, score_name].median(skipna=True)
        fixed.loc[scale_median_impute, score_name] = scale_median
        fixed[f"{score_name}_Missing_Items"] = missing_count
        fixed[f"{score_name}_Imputed_Flag"] = np.where(eligible, "Person-mean imputed", "")
        fixed.loc[scale_median_impute, f"{score_name}_Imputed_Flag"] = "Scale-median imputed; source had >20% missing items"

        if int(eligible.sum()) or int(scale_median_impute.sum()):
            imputation_rows.append(
                {
                    "Scale": score_name,
                    "Label": label,
                    "Items": len(item_cols),
                    "Allowed missing items": threshold,
                    "Rows person-mean imputed": int(eligible.sum()),
                    "Rows scale-median imputed": int(scale_median_impute.sum()),
                    "Rule": "Person-mean imputation when <=20% of items are missing; scale median when more items are missing.",
                }
            )

    for col in fixed.columns[:79]:
        if fixed[col].dtype == "object":
            fixed[col] = fixed[col].fillna("Missing in source")

    fixed["PHQ9_Category"] = pd.cut(
        fixed["PHQ9_Total"],
        bins=[-0.01, 4, 9, 14, 19, 27],
        labels=["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"],
    )
    fixed["PHQ9_ModeratePlus"] = np.where(fixed["PHQ9_Total"] >= 10, "Yes", "No")
    fixed.loc[fixed["PHQ9_Total"].isna(), "PHQ9_ModeratePlus"] = np.nan
    fixed["GAD7_Category"] = pd.cut(
        fixed["GAD7_Total"],
        bins=[-0.01, 4, 9, 14, 21],
        labels=["Minimal", "Mild", "Moderate", "Severe"],
    )
    fixed["GAD7_ModeratePlus"] = np.where(fixed["GAD7_Total"] >= 10, "Yes", "No")
    fixed.loc[fixed["GAD7_Total"].isna(), "GAD7_ModeratePlus"] = np.nan

    total_imputed = sum(r["Rows person-mean imputed"] for r in imputation_rows)
    total_scale_median = sum(r["Rows scale-median imputed"] for r in imputation_rows)
    fix_rows.append(
        {
            "Issue": "Missing item-level responses",
            "Fix applied in workbook": "Created analysis-ready scale scores using transparent imputation.",
            "Changes made": f"{total_imputed} scale scores person-mean imputed; {total_scale_median} scale scores scale-median imputed.",
            "Interpretation note": "Original scores are retained in *_Original columns; imputation details are in Missing_Value_Actions.",
        }
    )

    return fixed, pd.DataFrame(fix_rows), pd.DataFrame(imputation_rows)


def cronbach_alpha(frame):
    clean = frame.dropna()
    if clean.shape[0] < 3 or clean.shape[1] < 2:
        return np.nan
    item_variances = clean.var(axis=0, ddof=1)
    total_variance = clean.sum(axis=1).var(ddof=1)
    if total_variance == 0 or pd.isna(total_variance):
        return np.nan
    k = clean.shape[1]
    return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)


def p_label(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def effect_size_label(value):
    if pd.isna(value):
        return "Not available"
    a = abs(value)
    if a >= 0.5:
        return "Large"
    if a >= 0.3:
        return "Moderate"
    if a >= 0.1:
        return "Small"
    return "Very small"


def build_audit_tables(df_raw, scored, raw_info):
    duplicate_rows = int(df_raw.duplicated().sum())
    audit_rows = [
        ("Rows", raw_info["n_data_rows"], "Source CSV data rows"),
        ("Columns", raw_info["n_columns"], "Source CSV columns"),
        ("Duplicate raw headers", len(raw_info["duplicate_raw"]), "; ".join(raw_info["duplicate_raw"]) or "None"),
        ("Duplicate trimmed headers", len(raw_info["duplicate_trimmed"]), "; ".join(raw_info["duplicate_trimmed"]) or "None"),
        ("Blank headers", len(raw_info["blank_headers"]), ", ".join(map(str, raw_info["blank_headers"])) or "None"),
        ("Headers with leading/trailing spaces", len(raw_info["headers_with_edge_spaces"]), "; ".join(raw_info["headers_with_edge_spaces"]) or "None"),
        ("Rows with unexpected column counts", len(raw_info["bad_row_lengths"]), "See Row_Length_Issues sheet if nonzero"),
        ("Fully duplicated rows", duplicate_rows, "Exact duplicate data rows"),
        ("Unparseable Age values", int(scored["Age"].isna().sum()), "Age converted with pandas.to_numeric"),
        ("Unparseable Timestamp values", int(scored["Timestamp"].isna().sum()), "Timestamp parsed day-first"),
        ("CSV formulas detected", int(df_raw.astype(str).apply(lambda col: col.str.startswith("=", na=False)).sum().sum()), "CSV has text values only; formulas would be literal text"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["Check", "Issue count / value", "Details"])

    missing = (
        scored[df_raw.columns]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing count"})
    )
    missing["Missing percent"] = (missing["Missing count"] / len(scored) * 100).round(2)
    missing = missing.sort_values(["Missing count", "Column"], ascending=[False, True])

    format_rows = []
    for col in df_raw.columns:
        original = df_raw[col]
        stripped = original.map(norm_text)
        changed = int((original.fillna("").astype(str) != stripped.fillna("").astype(str)).sum())
        if changed:
            format_rows.append({"Column": col, "Values changed by trim/collapse whitespace": changed})
    format_issues = pd.DataFrame(format_rows)

    return audit, missing, pd.DataFrame(raw_info["bad_row_lengths"]), format_issues


def unknown_response_table(scored):
    rows = []
    score_cols = [c for c in scored.columns if c.startswith("item_") and c.endswith("_score")]
    for score_col in score_cols:
        index = int(score_col.split("_")[1])
        source_col = scored.columns[index]
        bad = scored[source_col].notna() & scored[score_col].isna()
        for value, count in scored.loc[bad, source_col].value_counts().items():
            rows.append({"Column": source_col, "Value not scored": value, "Count": int(count)})
    return pd.DataFrame(rows)


def descriptive_tables(scored):
    available = [c for c in PRIMARY_CONTINUOUS if c in scored.columns]
    desc = scored[available].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    desc = desc.rename(columns={"index": "Variable", "50%": "median", "25%": "p25", "75%": "p75"})
    desc["missing"] = [int(scored[c].isna().sum()) for c in desc["Variable"]]
    desc["sem"] = [scored[c].dropna().sem() for c in desc["Variable"]]
    desc["95% CI low"] = [
        scored[c].dropna().mean() - stats.t.ppf(0.975, len(scored[c].dropna()) - 1) * scored[c].dropna().sem()
        if len(scored[c].dropna()) > 1
        else np.nan
        for c in desc["Variable"]
    ]
    desc["95% CI high"] = [
        scored[c].dropna().mean() + stats.t.ppf(0.975, len(scored[c].dropna()) - 1) * scored[c].dropna().sem()
        if len(scored[c].dropna()) > 1
        else np.nan
        for c in desc["Variable"]
    ]
    desc["IQR"] = desc["p75"] - desc["p25"]
    desc["skewness"] = [stats.skew(scored[c].dropna(), bias=False) if len(scored[c].dropna()) >= 3 else np.nan for c in desc["Variable"]]
    desc["kurtosis"] = [stats.kurtosis(scored[c].dropna(), bias=False) if len(scored[c].dropna()) >= 4 else np.nan for c in desc["Variable"]]
    desc = desc[
        [
            "Variable",
            "count",
            "missing",
            "mean",
            "sem",
            "95% CI low",
            "95% CI high",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "IQR",
            "max",
            "skewness",
            "kurtosis",
        ]
    ].round(3)

    normality_rows = []
    for col in available:
        values = scored[col].dropna()
        if 3 <= len(values) <= 5000:
            stat, p = stats.shapiro(values)
            normality_rows.append({"Variable": col, "N": len(values), "Shapiro W": stat, "p-value": p, "Flag": "Non-normal" if p < 0.05 else "No strong departure"})
    normality = pd.DataFrame(normality_rows).round(4)

    freq_rows = []
    for col in ["Gender", "Specialization", "Designation", "PHQ9_Category", "GAD7_Category", "PHQ9_ModeratePlus", "GAD7_ModeratePlus"]:
        if col not in scored.columns:
            continue
        counts = scored[col].value_counts(dropna=False)
        for value, count in counts.items():
            freq_rows.append({"Variable": col, "Level": str(value), "Count": int(count), "Percent": round(count / len(scored) * 100, 2)})
    frequencies = pd.DataFrame(freq_rows)
    return desc, normality, frequencies


def reliability_table(scored):
    rows = []
    for score_name, label, indices, _, _ in SCALE_DEFINITIONS:
        item_cols = [f"item_{i:02d}_score" for i in indices if f"item_{i:02d}_score" in scored.columns]
        rows.append(
            {
                "Scale": score_name,
                "Label": label,
                "Items": len(item_cols),
                "Complete responses": int(scored[item_cols].dropna().shape[0]) if item_cols else 0,
                "Cronbach alpha": cronbach_alpha(scored[item_cols]) if item_cols else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def correlation_tables(scored):
    variables = [c for c in PRIMARY_CONTINUOUS if c in scored.columns]
    pearson_matrix = scored[variables].corr(method="pearson").round(3)
    spearman_matrix = scored[variables].corr(method="spearman").round(3)

    rows = []
    for i, var1 in enumerate(variables):
        for var2 in variables[i + 1 :]:
            common = scored[[var1, var2]].dropna()
            if len(common) >= 3:
                r, p = stats.pearsonr(common[var1], common[var2])
                rho, sp = stats.spearmanr(common[var1], common[var2])
            else:
                r = p = rho = sp = np.nan
            rows.append(
                {
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "N": len(common),
                    "Pearson r": r,
                    "Pearson p": p,
                    "Spearman rho": rho,
                    "Spearman p": sp,
                    "Effect size": effect_size_label(r),
                    "Significance": p_label(p),
                }
            )
    long_corr = pd.DataFrame(rows).sort_values(["Pearson p", "Pearson r"], ascending=[True, False]).round(4)
    if not long_corr.empty:
        pearson_mask = long_corr["Pearson p"].notna()
        spearman_mask = long_corr["Spearman p"].notna()
        long_corr.loc[pearson_mask, "Pearson q (BH-FDR)"] = multipletests(long_corr.loc[pearson_mask, "Pearson p"], method="fdr_bh")[1]
        long_corr.loc[spearman_mask, "Spearman q (BH-FDR)"] = multipletests(long_corr.loc[spearman_mask, "Spearman p"], method="fdr_bh")[1]
        long_corr["Pearson FDR significant"] = np.where(long_corr["Pearson q (BH-FDR)"] < 0.05, "Yes", "No")
        long_corr = long_corr.round(4)
    return pearson_matrix, spearman_matrix, long_corr


def association_tests(scored):
    rows = []
    grouping_columns = [c for c in ["Gender", "Designation", "Specialization"] if c in scored.columns]
    outcomes = [c for c in PRIMARY_CONTINUOUS if c in scored.columns and c != "Age"]

    for group_col in grouping_columns:
        groups = scored[group_col].dropna().value_counts()
        valid_levels = list(groups[groups >= 2].index)
        if len(valid_levels) < 2:
            continue
        for outcome in outcomes:
            data = scored[[group_col, outcome]].dropna()
            data = data[data[group_col].isin(valid_levels)]
            split = [g[outcome].dropna() for _, g in data.groupby(group_col)]
            split = [s for s in split if len(s) >= 2]
            if len(split) < 2:
                continue

            if len(split) == 2:
                stat, p = stats.mannwhitneyu(split[0], split[1], alternative="two-sided")
                test = "Mann-Whitney U"
                n1, n2 = len(split[0]), len(split[1])
                z = (stat - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                effect = abs(z) / np.sqrt(n1 + n2)
                effect_name = "rank-biserial approx r"
            else:
                stat, p = stats.kruskal(*split)
                test = "Kruskal-Wallis"
                n_total = sum(len(s) for s in split)
                effect = max((stat - len(split) + 1) / (n_total - len(split)), 0) if n_total > len(split) else np.nan
                effect_name = "epsilon squared"
            rows.append(
                {
                    "Predictor": group_col,
                    "Outcome": outcome,
                    "Test": test,
                    "Groups used": len(split),
                    "N": int(sum(len(s) for s in split)),
                    "Statistic": stat,
                    "Effect size": effect,
                    "Effect size type": effect_name,
                    "p-value": p,
                    "Significance": p_label(p),
                }
            )

    for cat_outcome in ["PHQ9_Category", "GAD7_Category", "PHQ9_ModeratePlus", "GAD7_ModeratePlus"]:
        if cat_outcome not in scored.columns:
            continue
        for group_col in grouping_columns:
            table = pd.crosstab(scored[group_col], scored[cat_outcome])
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(table)
                rows.append(
                    {
                        "Predictor": group_col,
                        "Outcome": cat_outcome,
                        "Test": "Chi-square",
                        "Groups used": table.shape[0],
                        "N": int(table.to_numpy().sum()),
                        "Statistic": chi2,
                        "Effect size": np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))),
                        "Effect size type": "Cramer's V",
                        "p-value": p,
                        "Significance": p_label(p),
                    }
                )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("p-value")
    mask = result["p-value"].notna()
    result.loc[mask, "q-value (BH-FDR)"] = multipletests(result.loc[mask, "p-value"], method="fdr_bh")[1]
    result["FDR significant"] = np.where(result["q-value (BH-FDR)"] < 0.05, "Yes", "No")
    return result.round(4)


def group_summary_table(scored):
    rows = []
    outcomes = [c for c in PRIMARY_CONTINUOUS if c in scored.columns and c != "Age"]
    for group_col in [c for c in ["Gender", "Designation", "Specialization"] if c in scored.columns]:
        for level, group in scored.groupby(group_col, dropna=False):
            if len(group) < 2:
                continue
            for outcome in outcomes:
                values = group[outcome].dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "Group variable": group_col,
                        "Level": level,
                        "Outcome": outcome,
                        "N": len(values),
                        "Mean": values.mean(),
                        "SD": values.std(),
                        "Median": values.median(),
                        "IQR": values.quantile(0.75) - values.quantile(0.25),
                        "Min": values.min(),
                        "Max": values.max(),
                    }
                )
    return pd.DataFrame(rows).round(4)


def outlier_screen(scored):
    rows = []
    for col in [c for c in PRIMARY_CONTINUOUS if c in scored.columns]:
        values = scored[col].dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mean = values.mean()
        sd = values.std()
        z = (values - mean) / sd if sd and not pd.isna(sd) else pd.Series(np.nan, index=values.index)
        rows.append(
            {
                "Variable": col,
                "N": len(values),
                "IQR lower fence": lower,
                "IQR upper fence": upper,
                "IQR outlier count": int(((values < lower) | (values > upper)).sum()),
                "|z| > 3 count": int((z.abs() > 3).sum()),
                "Min": values.min(),
                "Max": values.max(),
            }
        )
    return pd.DataFrame(rows).round(4)


def prepare_regression_data(scored, outcome, predictors):
    cols = [outcome] + predictors
    data = scored[cols].copy()
    data = data.dropna()
    y = pd.to_numeric(data[outcome], errors="coerce")
    x = data[predictors].copy()

    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].astype("category")

    x = pd.get_dummies(x, drop_first=True, dtype=float)
    x = x.apply(pd.to_numeric, errors="coerce")
    combined = pd.concat([y.rename(outcome), x], axis=1).dropna()
    y = combined[outcome]
    x = combined.drop(columns=[outcome])

    nunique = x.nunique(dropna=True)
    x = x.loc[:, nunique > 1]
    return y, x


def standardized_betas(y, x):
    if x.empty or y.std(ddof=0) == 0:
        return pd.Series(dtype=float)
    x_std = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    y_std = (y - y.mean()) / y.std(ddof=0)
    valid = x_std.dropna(axis=1)
    if valid.empty:
        return pd.Series(dtype=float)
    model = sm.OLS(y_std, sm.add_constant(valid)).fit()
    return model.params.drop("const", errors="ignore")


def vif_table(x):
    if x.shape[1] < 2:
        return pd.DataFrame({"Predictor": list(x.columns), "VIF": [np.nan] * x.shape[1]})
    x_const = sm.add_constant(x)
    rows = []
    for i, col in enumerate(x_const.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(x_const.values, i)
        except Exception:
            vif = np.nan
        rows.append({"Predictor": col, "VIF": vif})
    return pd.DataFrame(rows)


def run_regressions(scored):
    model_specs = [
        {
            "Model": "Predict PHQ-9 depression",
            "Outcome": "PHQ9_Total",
            "Predictors": ["Age", "Gender", "SV_Emotional_Total", "Peer_Support_Total", "Supervisor_Support_Total", "Org_Support_Total", "Burnout_Total"],
        },
        {
            "Model": "Predict GAD-7 anxiety",
            "Outcome": "GAD7_Total",
            "Predictors": ["Age", "Gender", "SV_Emotional_Total", "Peer_Support_Total", "Supervisor_Support_Total", "Org_Support_Total", "Burnout_Total"],
        },
        {
            "Model": "Predict burnout",
            "Outcome": "Burnout_Total",
            "Predictors": ["Age", "Gender", "PHQ9_Total", "GAD7_Total", "SV_Emotional_Total", "Org_Support_Total", "Professional_Growth_Total"],
        },
        {
            "Model": "Predict intention to leave",
            "Outcome": "Intent_Leave_Total",
            "Predictors": ["Age", "Gender", "PHQ9_Total", "GAD7_Total", "SV_Emotional_Total", "Org_Support_Total", "Performance_Impact_Total", "Burnout_Total"],
        },
        {
            "Model": "Predict performance impact",
            "Outcome": "Performance_Impact_Total",
            "Predictors": ["Age", "Gender", "PHQ9_Total", "GAD7_Total", "SV_Emotional_Total", "Supervisor_Support_Total", "Org_Support_Total", "Burnout_Total"],
        },
    ]

    summary_rows = []
    coef_rows = []
    diagnostic_rows = []
    vif_rows = []

    for spec in model_specs:
        predictors = [p for p in spec["Predictors"] if p in scored.columns]
        y, x = prepare_regression_data(scored, spec["Outcome"], predictors)
        if len(y) < max(10, x.shape[1] + 3) or x.empty:
            summary_rows.append(
                {
                    "Model": spec["Model"],
                    "Outcome": spec["Outcome"],
                    "N": len(y),
                    "Predictors": x.shape[1],
                    "Status": "Skipped: insufficient complete observations or no usable predictors",
                }
            )
            continue

        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        robust_model = model.get_robustcov_results(cov_type="HC3")
        std_betas = standardized_betas(y, x)
        predictions = model.predict(x_const)
        residuals = model.resid
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))

        summary_rows.append(
            {
                "Model": spec["Model"],
                "Outcome": spec["Outcome"],
                "N": int(model.nobs),
                "Predictors": x.shape[1],
                "R-squared": model.rsquared,
                "Adjusted R-squared": model.rsquared_adj,
                "F-statistic": model.fvalue,
                "Model p-value": model.f_pvalue,
                "RMSE": rmse,
                "MAE": mae,
                "AIC": model.aic,
                "BIC": model.bic,
                "Status": "Fit; use HC3 robust coefficient columns for inference",
            }
        )

        conf = model.conf_int()
        robust_conf = pd.DataFrame(robust_model.conf_int(), index=model.params.index)
        robust_bse = pd.Series(robust_model.bse, index=model.params.index)
        robust_t = pd.Series(robust_model.tvalues, index=model.params.index)
        robust_p = pd.Series(robust_model.pvalues, index=model.params.index)
        for term in model.params.index:
            coef_rows.append(
                {
                    "Model": spec["Model"],
                    "Outcome": spec["Outcome"],
                    "Term": term,
                    "B": model.params[term],
                    "Std error": model.bse[term],
                    "t": model.tvalues[term],
                    "p-value": model.pvalues[term],
                    "95% CI low": conf.loc[term, 0],
                    "95% CI high": conf.loc[term, 1],
                    "HC3 robust std error": robust_bse[term],
                    "HC3 robust t": robust_t[term],
                    "HC3 robust p-value": robust_p[term],
                    "HC3 robust 95% CI low": robust_conf.loc[term, 0],
                    "HC3 robust 95% CI high": robust_conf.loc[term, 1],
                    "Standardized beta": std_betas.get(term, np.nan),
                    "OLS significance": p_label(model.pvalues[term]),
                    "HC3 robust significance": p_label(robust_p[term]),
                }
            )

        shap_w, shap_p = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 else (np.nan, np.nan)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        diagnostic_rows.append(
            {
                "Model": spec["Model"],
                "Residual Shapiro W": shap_w,
                "Residual normality p": shap_p,
                "Breusch-Pagan statistic": bp_stat,
                "Breusch-Pagan p": bp_p,
                "Max Cook distance": model.get_influence().cooks_distance[0].max(),
                "Inference fix applied": "HC3 robust standard errors added",
                "Recommended p-value column": "HC3 robust p-value",
            }
        )

        vt = vif_table(x)
        vt.insert(0, "Model", spec["Model"])
        vif_rows.extend(vt.to_dict("records"))

    return (
        pd.DataFrame(summary_rows).round(4),
        pd.DataFrame(coef_rows).sort_values(["Model", "p-value"]).round(4) if coef_rows else pd.DataFrame(),
        pd.DataFrame(diagnostic_rows).round(4),
        pd.DataFrame(vif_rows).round(4),
    )


def prioritized_issues(audit, missing, unknown, format_issues, regression_diagnostics, vif):
    rows = []

    def add(priority, issue, evidence, action, status="Open / monitor"):
        rows.append({"Priority": priority, "Status": status, "Issue": issue, "Evidence": evidence, "Suggested action": action})

    header_space = int(audit.loc[audit["Check"] == "Headers with leading/trailing spaces", "Issue count / value"].iloc[0])
    if header_space:
        add(
            "High",
            "Headers contain leading/trailing spaces in source CSV",
            f"{header_space} headers",
            "Fixed in workbook outputs using trimmed/collapsed headers; source CSV left unchanged.",
            "Fixed in Excel output",
        )

    if not missing.empty and missing["Missing count"].max() > 0:
        top_missing = missing[missing["Missing count"] > 0].head(5)
        evidence = "; ".join(f"{r.Column}: {int(r['Missing count'])}" for _, r in top_missing.iterrows())
        add(
            "High",
            "Missing values are present in source CSV",
            evidence,
            "Fixed in analysis-ready data: categorical blanks are labelled; scale scores use person-mean imputation for small gaps and scale-median imputation for larger gaps.",
            "Fixed in Excel output with flags",
        )

    if unknown is not None and not unknown.empty:
        add("High", "Some survey responses could not be scored", f"{len(unknown)} unrecognized value entries", "Review Unknown_Response_Values before interpreting scale scores.")

    if format_issues is not None and not format_issues.empty:
        top = format_issues.head(5)
        evidence = "; ".join(f"{r.Column}: {int(r['Values changed by trim/collapse whitespace'])}" for _, r in top.iterrows())
        add("Medium", "Text values required whitespace cleanup", evidence, "Continue normalizing categorical values before tests.")

    if "Age" in missing["Column"].values:
        age_missing = int(missing.loc[missing["Column"] == "Age", "Missing count"].iloc[0])
        if age_missing:
            add("Medium", "Age has missing or nonnumeric values", f"{age_missing} records", "Exclude or impute explicitly for adjusted models.")

    if regression_diagnostics is not None and not regression_diagnostics.empty:
        non_normal = regression_diagnostics[regression_diagnostics["Residual normality p"] < 0.05]
        hetero = regression_diagnostics[regression_diagnostics["Breusch-Pagan p"] < 0.05]
        if not non_normal.empty:
            add(
                "Medium",
                "Regression residuals may be non-normal",
                f"{len(non_normal)} model(s)",
                "HC3 robust standard errors and p-values added; use HC3 columns for inference.",
                "Fixed in Excel output",
            )
        if not hetero.empty:
            add(
                "Medium",
                "Regression heteroskedasticity detected",
                f"{len(hetero)} model(s)",
                "HC3 robust standard errors and p-values added; use HC3 columns for inference.",
                "Fixed in Excel output",
            )

    if vif is not None and not vif.empty and vif["VIF"].dropna().gt(5).any():
        high_vif = vif[vif["VIF"] > 5].sort_values("VIF", ascending=False).head(5)
        evidence = "; ".join(f"{r.Predictor}: {r.VIF:.1f}" for _, r in high_vif.iterrows())
        add("Medium", "Multicollinearity risk in regression predictors", evidence, "Remove overlapping predictors or present models as exploratory.")

    dup_count = int(audit.loc[audit["Check"] == "Fully duplicated rows", "Issue count / value"].iloc[0])
    if dup_count:
        add("Low", "Exact duplicate rows detected", f"{dup_count} duplicate row(s)", "Verify whether duplicate submissions should be removed.")

    if not rows:
        add("Low", "No major data integrity issues detected", "Audit checks passed", "Proceed with statistical interpretation using reported assumptions.")
    return pd.DataFrame(rows)


def make_executive_insights(scored, descriptives, correlation_long, associations, group_summaries, reg_summary, reg_coef):
    rows = []

    def add(section, insight, evidence, interpretation):
        rows.append(
            {
                "Section": section,
                "Insight": insight,
                "Evidence from cleaned data": evidence,
                "Interpretation / reporting note": interpretation,
            }
        )

    n = len(scored)
    age_mean = scored["Age"].mean()
    age_sd = scored["Age"].std()
    female_pct = (scored["Gender"].eq("Female").mean() * 100) if "Gender" in scored.columns else np.nan
    male_pct = (scored["Gender"].eq("Male").mean() * 100) if "Gender" in scored.columns else np.nan
    add(
        "Sample profile",
        "The cleaned dataset contains 99 respondent records ready for statistical analysis.",
        f"N={n}; age mean={age_mean:.1f} years, SD={age_sd:.1f}; female={female_pct:.1f}%, male={male_pct:.1f}%.",
        "Use the cleaned analysis-ready dataset for all reported statistics.",
    )

    if "Designation" in scored.columns:
        top_designations = scored["Designation"].value_counts().head(3)
        add(
            "Sample profile",
            "The largest respondent groups are concentrated in a few designations.",
            "; ".join(f"{idx}: {val}" for idx, val in top_designations.items()),
            "When comparing groups, interpret small designation groups cautiously.",
        )

    phq_mod = (scored["PHQ9_Total"] >= 10).mean() * 100
    gad_mod = (scored["GAD7_Total"] >= 10).mean() * 100
    add(
        "Mental health screening",
        "Moderate-or-higher depression/anxiety symptom burden is present in a minority of responses.",
        f"PHQ-9 >=10: {phq_mod:.1f}%; GAD-7 >=10: {gad_mod:.1f}%.",
        "These are screening thresholds, not diagnoses; report as symptom burden indicators.",
    )

    key_means = descriptives.set_index("Variable") if not descriptives.empty else pd.DataFrame()
    for var, label in [
        ("SV_Emotional_Total", "second victim emotional impact"),
        ("Resource_Need_Total", "need for support resources"),
        ("Peer_Support_Total", "peer support"),
        ("Supervisor_Support_Total", "supervisor support"),
        ("Org_Support_Total", "organisational support"),
        ("Burnout_Total", "burnout"),
        ("Intent_Leave_Total", "intention to leave"),
    ]:
        if var in key_means.index:
            add(
                "Scale summary",
                f"Average {label} score is {key_means.loc[var, 'mean']:.2f}.",
                f"Mean={key_means.loc[var, 'mean']:.2f}; median={key_means.loc[var, 'median']:.2f}; IQR={key_means.loc[var, 'IQR']:.2f}.",
                "Most non-PHQ/GAD scales are 1-5 averages; higher values reflect more of the named construct.",
            )

    if correlation_long is not None and not correlation_long.empty:
        corr = correlation_long.copy()
        corr = corr[corr["Variable 1"] != corr["Variable 2"]]
        corr = corr[corr["Pearson p"] < 0.05].head(8)
        for _, row in corr.iterrows():
            add(
                "Correlation",
                f"{row['Variable 1']} is associated with {row['Variable 2']}.",
                f"Pearson r={row['Pearson r']:.3f}; p={row['Pearson p']:.4f}; Spearman rho={row['Spearman rho']:.3f}; effect={row['Effect size']}.",
                "Correlation does not prove causation, but it identifies paired outcomes that move together in this dataset.",
            )

    if reg_coef is not None and not reg_coef.empty:
        coefs = reg_coef[(reg_coef["Term"] != "const") & (reg_coef["HC3 robust p-value"] < 0.05)].copy()
        coefs = coefs.sort_values(["Outcome", "HC3 robust p-value"])
        for _, row in coefs.head(12).iterrows():
            direction = "higher" if row["B"] > 0 else "lower"
            add(
                "Linear regression",
                f"{row['Term']} is an independent predictor of {row['Outcome']}.",
                f"B={row['B']:.3f}; HC3 robust p={row['HC3 robust p-value']:.4f}; 95% CI [{row['HC3 robust 95% CI low']:.3f}, {row['HC3 robust 95% CI high']:.3f}].",
                f"After adjustment for other model predictors, higher {row['Term']} is linked with {direction} {row['Outcome']}.",
            )

    if reg_summary is not None and not reg_summary.empty:
        best_models = reg_summary.sort_values("Adjusted R-squared", ascending=False).head(3)
        for _, row in best_models.iterrows():
            add(
                "Linear regression",
                f"The model for {row['Outcome']} explains a meaningful share of variance.",
                f"Adjusted R-squared={row['Adjusted R-squared']:.3f}; N={int(row['N'])}; predictors={int(row['Predictors'])}; model p={row['Model p-value']:.4f}.",
                "Use Regression_Coefficients for predictor-level interpretation and HC3 robust p-values.",
            )

    if associations is not None and not associations.empty:
        assoc = associations.sort_values("p-value").head(10)
        for _, row in assoc.iterrows():
            fdr = row.get("FDR significant", "No")
            add(
                "Association test",
                f"{row['Predictor']} differs by or is associated with {row['Outcome']}.",
                f"{row['Test']}; statistic={row['Statistic']:.3f}; p={row['p-value']:.4f}; effect size={row['Effect size']:.3f}; FDR significant={fdr}.",
                "Use group summaries to describe which subgroup has higher or lower scores.",
            )

    if group_summaries is not None and not group_summaries.empty:
        key_outcomes = ["PHQ9_Total", "GAD7_Total", "Burnout_Total", "Intent_Leave_Total", "Performance_Impact_Total"]
        for outcome in key_outcomes:
            gs = group_summaries[(group_summaries["Outcome"] == outcome) & (group_summaries["N"] >= 3)].copy()
            if gs.empty:
                continue
            overall = scored[outcome].mean()
            gs["Difference vs overall"] = gs["Mean"] - overall
            top = gs.sort_values("Difference vs overall", ascending=False).head(1).iloc[0]
            add(
                "Subgroup pattern",
                f"{top['Group variable']}={top['Level']} has the highest observed mean for {outcome}.",
                f"N={int(top['N'])}; mean={top['Mean']:.2f}; overall mean={overall:.2f}; difference={top['Difference vs overall']:.2f}.",
                "Treat subgroup findings as descriptive when group sizes are small.",
            )

    return pd.DataFrame(rows)


def label_for(name):
    if pd.isna(name):
        return ""
    text = str(name)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]
    for prefix, label in [("Gender_", "gender"), ("Designation_", "designation"), ("Specialization_", "specialization")]:
        if text.startswith(prefix):
            return f"{label}: {text.removeprefix(prefix)}"
    return text.replace("_Total", "").replace("_", " ").lower()


def format_p(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def relation_direction(value):
    if pd.isna(value):
        return "not available"
    if value > 0:
        return "increases together"
    if value < 0:
        return "inverse relationship"
    return "no direction"


def relationship_sentence(left, right, value, source="correlation"):
    left_label = label_for(left)
    right_label = label_for(right)
    if pd.isna(value):
        return f"{left_label} has no estimated directional relationship with {right_label}."
    if value > 0:
        verb = "predicts higher" if source == "regression" else "is associated with higher"
        return f"Higher {left_label} {verb} {right_label}."
    if value < 0:
        verb = "predicts lower" if source == "regression" else "is associated with lower"
        return f"Higher {left_label} {verb} {right_label}."
    return f"{left_label} has no directional relationship with {right_label}."


def make_strong_correlations(correlation_long):
    if correlation_long is None or correlation_long.empty:
        return pd.DataFrame()
    corr = correlation_long.copy()
    corr["Abs Pearson r"] = corr["Pearson r"].abs()
    corr = corr[(corr["Pearson p"] < 0.05) & (corr["Abs Pearson r"] >= 0.3)]
    corr = corr.sort_values(["Abs Pearson r", "Pearson p"], ascending=[False, True])
    rows = []
    for _, row in corr.iterrows():
        rows.append(
            {
                "Insight": relationship_sentence(row["Variable 1"], row["Variable 2"], row["Pearson r"]),
                "Variable 1": label_for(row["Variable 1"]),
                "Variable 2": label_for(row["Variable 2"]),
                "Direction": relation_direction(row["Pearson r"]),
                "Pearson r": row["Pearson r"],
                "Pearson p": row["Pearson p"],
                "Pearson q (BH-FDR)": row.get("Pearson q (BH-FDR)", np.nan),
                "Spearman rho": row["Spearman rho"],
                "Spearman p": row["Spearman p"],
                "Effect size": row["Effect size"],
                "Reporting note": "Association only; this does not establish causation.",
            }
        )
    return pd.DataFrame(rows).round(4)


def make_regression_directions(reg_summary, reg_coef):
    if reg_coef is None or reg_coef.empty:
        return pd.DataFrame()
    summary = reg_summary.set_index("Model") if reg_summary is not None and not reg_summary.empty else pd.DataFrame()
    rows = []
    for _, row in reg_coef[reg_coef["Term"] != "const"].iterrows():
        model_info = summary.loc[row["Model"]] if row["Model"] in summary.index else {}
        p_value = row["HC3 robust p-value"]
        sig = "Yes" if not pd.isna(p_value) and p_value < 0.05 else "No"
        term = row["Term"]
        outcome = row["Outcome"]
        if "_" in str(term) and not str(term).endswith("_Total"):
            direction_word = "higher" if row["B"] > 0 else "lower"
            sentence = f"Compared with its reference group, {label_for(term)} is linked with {direction_word} {label_for(outcome)}."
        else:
            sentence = relationship_sentence(term, outcome, row["B"], source="regression")
        rows.append(
            {
                "Insight": sentence,
                "Model": row["Model"],
                "Outcome": label_for(outcome),
                "Predictor": label_for(term),
                "Direction": "outcome increases as predictor increases" if row["B"] > 0 else "outcome decreases as predictor increases",
                "Statistically significant": sig,
                "B coefficient": row["B"],
                "HC3 robust p-value": p_value,
                "HC3 robust 95% CI low": row["HC3 robust 95% CI low"],
                "HC3 robust 95% CI high": row["HC3 robust 95% CI high"],
                "Standardized beta": row.get("Standardized beta", np.nan),
                "Adjusted R-squared": model_info.get("Adjusted R-squared", np.nan) if hasattr(model_info, "get") else np.nan,
                "N": model_info.get("N", np.nan) if hasattr(model_info, "get") else np.nan,
                "Reporting note": "Adjusted linear regression; use HC3 robust p-values for inference.",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["Statistically significant", "HC3 robust p-value"], ascending=[False, True]).round(4)


def make_relationship_insights(scored, strong_correlations, regression_directions, associations, high_risk_segments):
    rows = []

    def add(category, insight, evidence, interpretation, priority):
        rows.append(
            {
                "Priority": priority,
                "Category": category,
                "Relationship insight": insight,
                "Evidence": evidence,
                "Interpretation": interpretation,
            }
        )

    if scored is not None and not scored.empty:
        add(
            "Dataset",
            "The cleaned dataset has complete values for the main analysis traits.",
            f"N={len(scored)}; PHQ-9 >=10: {(scored['PHQ9_Total'].ge(10).mean() * 100):.1f}%; GAD-7 >=10: {(scored['GAD7_Total'].ge(10).mean() * 100):.1f}%.",
            "Use this workbook for relationship-oriented reporting; findings are exploratory associations.",
            1,
        )

    if strong_correlations is not None and not strong_correlations.empty:
        for _, row in strong_correlations.head(10).iterrows():
            add(
                "Correlation",
                row["Insight"],
                f"Pearson r={row['Pearson r']:.3f}; p={format_p(row['Pearson p'])}; Spearman rho={row['Spearman rho']:.3f}; direction={row['Direction']}.",
                "These two traits move together in the cleaned data, but the relationship is not causal evidence.",
                2,
            )

    if regression_directions is not None and not regression_directions.empty:
        sig = regression_directions[regression_directions["Statistically significant"] == "Yes"].copy()
        for _, row in sig.head(12).iterrows():
            add(
                "Linear regression",
                row["Insight"],
                f"B={row['B coefficient']:.3f}; HC3 robust p={format_p(row['HC3 robust p-value'])}; 95% CI [{row['HC3 robust 95% CI low']:.3f}, {row['HC3 robust 95% CI high']:.3f}]; adjusted R2={row['Adjusted R-squared']:.3f}.",
                "This relationship remains after adjustment for the other predictors in that model.",
                3,
            )

    if associations is not None and not associations.empty:
        assoc = associations[associations["p-value"] < 0.05].sort_values("p-value").head(8)
        for _, row in assoc.iterrows():
            add(
                "Association test",
                f"{label_for(row['Predictor'])} is associated with differences in {label_for(row['Outcome'])}.",
                f"{row['Test']}; p={format_p(row['p-value'])}; effect size={row['Effect size']:.3f}; FDR significant={row.get('FDR significant', 'No')}.",
                "Use subgroup means in High_Risk_Segments before interpreting which group is elevated.",
                4,
            )

    if high_risk_segments is not None and not high_risk_segments.empty:
        for _, row in high_risk_segments.head(8).iterrows():
            add(
                "High-risk segment",
                f"{label_for(row['Segment variable'])}={row['Segment']} has elevated {label_for(row['Outcome'])}.",
                f"N={int(row['N'])}; segment mean={row['Segment mean']:.2f}; overall mean={row['Overall mean']:.2f}; difference={row['Difference vs overall']:.2f}.",
                "Descriptive segment insight; smaller groups should be interpreted cautiously.",
                5,
            )

    return pd.DataFrame(rows)


def make_high_risk_segments(scored):
    rows = []
    outcomes = [c for c in RELATIONSHIP_OUTCOMES if c in scored.columns]
    for group_col in [c for c in ["Gender", "Designation", "Specialization"] if c in scored.columns]:
        for outcome in outcomes:
            overall = scored[outcome].mean()
            for level, group in scored.groupby(group_col, dropna=False):
                if len(group) < 3:
                    continue
                mean = group[outcome].mean()
                rows.append(
                    {
                        "Segment variable": group_col,
                        "Segment": level,
                        "Outcome": outcome,
                        "N": len(group),
                        "Segment mean": mean,
                        "Overall mean": overall,
                        "Difference vs overall": mean - overall,
                        "Segment median": group[outcome].median(),
                        "Reporting note": "Descriptive; groups with small N should be interpreted cautiously.",
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Difference vs overall", ascending=False).round(4)


def make_scale_summary(descriptives):
    if descriptives is None or descriptives.empty:
        return pd.DataFrame()
    keep = [c for c in PRIMARY_CONTINUOUS if c in set(descriptives["Variable"])]
    result = descriptives[descriptives["Variable"].isin(keep)].copy()
    result["Trait"] = result["Variable"].map(label_for)
    result = result[
        [
            "Trait",
            "Variable",
            "count",
            "missing",
            "mean",
            "sem",
            "95% CI low",
            "95% CI high",
            "std",
            "median",
            "IQR",
            "min",
            "max",
            "skewness",
            "kurtosis",
        ]
    ]
    return result.round(4)


def make_methods_sheet():
    return pd.DataFrame(
        [
            {
                "Topic": "Dataset",
                "Method / interpretation": "The workbook uses the cleaned analysis-ready dataset generated from the source CSV. Main trait variables have missing values resolved using documented imputation rules.",
            },
            {
                "Topic": "Scoring",
                "Method / interpretation": "PHQ-9 and GAD-7 are summed. Other multi-item traits are averaged on a 1-5 scale. Reverse-coded support/burnout items are handled before scoring.",
            },
            {
                "Topic": "Correlations",
                "Method / interpretation": "Pearson and Spearman correlations describe how numeric traits move together. Positive values mean traits increase together; negative values mean inverse relationships.",
            },
            {
                "Topic": "Linear regression",
                "Method / interpretation": "OLS regression models use HC3 robust standard errors and p-values. Positive B means the outcome tends to increase as the predictor increases, after adjustment.",
            },
            {
                "Topic": "Associations",
                "Method / interpretation": "Mann-Whitney, Kruskal-Wallis, and Chi-square tests summarize subgroup differences or categorical associations.",
            },
            {
                "Topic": "Causality",
                "Method / interpretation": "The data are observational survey responses. Findings should be reported as associations or predictors, not proven causal effects.",
            },
        ]
    )


def write_relationship_workbook(output_dir, cleaned_data, descriptives, correlation_long, associations, reg_summary, reg_coef):
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "second_victim_relationship_insights.xlsx"
    strong_correlations = make_strong_correlations(correlation_long)
    regression_directions = make_regression_directions(reg_summary, reg_coef)
    high_risk_segments = make_high_risk_segments(cleaned_data)
    relationship_insights = make_relationship_insights(
        cleaned_data,
        strong_correlations,
        regression_directions,
        associations,
        high_risk_segments,
    )
    scale_summary = make_scale_summary(descriptives)
    methods = make_methods_sheet()

    tables = {
        "Relationship_Insights": relationship_insights,
        "Regression_Directions": regression_directions,
        "Strong_Correlations": strong_correlations,
        "Associations": associations,
        "High_Risk_Segments": high_risk_segments,
        "Scale_Summary": scale_summary,
        "Cleaned_Data": cleaned_data,
        "Methods": methods,
    }
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame({"Message": ["No rows generated"]})
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 65)

    for name, table in tables.items():
        if table is not None and not table.empty and name != "Cleaned_Data":
            table.to_csv(output_dir / f"{name.lower()}.csv", index=False)
    return workbook_path, relationship_insights


def write_outputs(output_dir, tables):
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "second_victim_statistical_analysis.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            if table is None or table.empty:
                pd.DataFrame({"Message": ["No rows generated"]}).to_excel(writer, sheet_name=safe_name, index=False)
            else:
                table.to_excel(writer, sheet_name=safe_name, index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)

    for name, table in tables.items():
        if table is not None and not table.empty and name in {
            "Executive_Insights",
            "Cleaned_Data",
            "Data_Cleaning_Log",
            "Fixes_Applied",
            "Missing_Value_Actions",
            "Descriptives",
            "Correlation_Long",
            "Associations",
            "Group_Summaries",
            "Outlier_Screen",
            "Regression_Summary",
            "Regression_Coefficients",
        }:
            table.to_csv(output_dir / f"{name.lower()}.csv", index=False)
    return workbook_path


def main():
    parser = argparse.ArgumentParser(description="Analyze Second Victim Phenomenon survey data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the source CSV file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for analysis outputs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    raw_info = inspect_raw_csv(input_path)
    raw_df = load_data(input_path, raw_info)
    scored, item_codebook, scale_metadata = add_scores(raw_df, raw_info)
    analysis_ready, fixes_applied, missing_value_actions = apply_analysis_fixes(scored)

    audit, missing, row_length_issues, format_issues = build_audit_tables(raw_df, scored, raw_info)
    unknown = unknown_response_table(scored)
    descriptives, normality, frequencies = descriptive_tables(analysis_ready)
    reliability = reliability_table(scored)
    pearson_matrix, spearman_matrix, correlation_long = correlation_tables(analysis_ready)
    associations = association_tests(analysis_ready)
    group_summaries = group_summary_table(analysis_ready)
    outliers = outlier_screen(analysis_ready)
    reg_summary, reg_coef, reg_diag, vif = run_regressions(analysis_ready)
    issues = prioritized_issues(audit, missing, unknown, format_issues, reg_diag, vif)
    executive_insights = make_executive_insights(
        analysis_ready,
        descriptives,
        correlation_long,
        associations,
        group_summaries,
        reg_summary,
        reg_coef,
    )

    scored_export_cols = list(raw_df.columns) + [c for c in scored.columns if c not in raw_df.columns]
    analysis_export_cols = list(raw_df.columns) + [c for c in analysis_ready.columns if c not in raw_df.columns]
    cleaned_data = analysis_ready[analysis_export_cols]
    tables = {
        "Executive_Insights": executive_insights,
        "Cleaned_Data": cleaned_data,
        "Descriptives": descriptives,
        "Normality": normality,
        "Frequencies": frequencies,
        "Reliability": reliability,
        "Pearson_Matrix": pearson_matrix.reset_index().rename(columns={"index": "Variable"}),
        "Spearman_Matrix": spearman_matrix.reset_index().rename(columns={"index": "Variable"}),
        "Correlation_Long": correlation_long,
        "Associations": associations,
        "Group_Summaries": group_summaries,
        "Outlier_Screen": outliers,
        "Regression_Summary": reg_summary,
        "Regression_Coefficients": reg_coef,
        "Regression_Diagnostics": reg_diag,
        "Regression_VIF": vif,
        "Scale_Metadata": scale_metadata,
        "Item_Codebook": item_codebook,
        "Data_Cleaning_Log": issues,
        "Fixes_Applied": fixes_applied,
        "Missingness": missing,
        "Missing_Value_Actions": missing_value_actions,
        "Audit": audit,
        "Row_Length_Issues": row_length_issues,
        "Format_Issues": format_issues,
        "Unknown_Response_Values": unknown,
        "Scored_Data_AnalysisReady": cleaned_data,
        "Scored_Data_Source": scored[scored_export_cols],
    }
    workbook_path = write_outputs(output_dir, tables)
    cleaned_csv_path = output_dir / "second_victim_cleaned_analysis_ready.csv"
    cleaned_xlsx_path = output_dir / "second_victim_cleaned_analysis_ready.xlsx"
    cleaned_data.to_csv(cleaned_csv_path, index=False)
    with pd.ExcelWriter(cleaned_xlsx_path, engine="openpyxl") as writer:
        cleaned_data.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        fixes_applied.to_excel(writer, sheet_name="Cleaning_Notes", index=False)
        missing_value_actions.to_excel(writer, sheet_name="Imputation_Rules", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)
    relationship_data = pd.read_csv(cleaned_csv_path)
    relationship_workbook_path, relationship_insights = write_relationship_workbook(
        output_dir,
        relationship_data,
        descriptives,
        correlation_long,
        associations,
        reg_summary,
        reg_coef,
    )

    print("\nAnalysis complete.")
    print(f"Source CSV: {input_path}")
    print(f"Rows analyzed: {len(scored)}")
    print(f"Output workbook: {workbook_path}")
    print(f"Relationship insights workbook: {relationship_workbook_path}")
    print(f"Cleaned dataset CSV: {cleaned_csv_path}")
    print(f"Cleaned dataset Excel: {cleaned_xlsx_path}")
    print("\nTop relationship insights:")
    print(relationship_insights.head(12).to_string(index=False))

    sig_corr = correlation_long[correlation_long["Pearson p"] < 0.05] if not correlation_long.empty else pd.DataFrame()
    sig_coef = reg_coef[(reg_coef["HC3 robust p-value"] < 0.05) & (reg_coef["Term"] != "const")] if not reg_coef.empty else pd.DataFrame()
    print(f"\nSignificant Pearson correlations: {len(sig_corr)}")
    print(f"Significant non-intercept regression coefficients using HC3 robust p-values: {len(sig_coef)}")

    print(
        dedent(
            """

            Interpretation notes:
            - PHQ-9 and GAD-7 are summed as 0-27 and 0-21 respectively.
            - Other multi-item scales are averaged on 1-5 scoring.
            - Negatively worded support and positively worded burnout items are reverse-coded.
            - Missing item scores use person-mean imputation for <=20% missing within a scale and scale-median imputation for larger gaps; original scores are retained in *_Original columns.
            - Regression models use ordinary least squares with HC3 robust standard errors for inference.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
