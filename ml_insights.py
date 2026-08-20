#!/usr/bin/env python3
"""
Machine-learning insights workbook for the Second Victim Phenomenon survey.

This script uses the cleaned analysis-ready dataset generated from analysis.py,
trains exploratory ML models, and writes an insights-focused Excel workbook.
The source CSV is not modified.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    DEFAULT_INPUT,
    add_scores,
    apply_analysis_fixes,
    inspect_raw_csv,
    load_data,
)

try:
    from sklearn.base import clone
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    print(
        "Missing scikit-learn. Run:\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install scikit-learn",
        flush=True,
    )
    raise exc


warnings.filterwarnings("ignore")

OUTPUT_DIR = "analysis_outputs"
OUTPUT_FILE = "second_victim_ml_insights.xlsx"

DEMOGRAPHIC_FEATURES = ["Age", "Gender", "Designation", "Specialization"]
SCALE_FEATURES = [
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

REGRESSION_OUTCOMES = [
    "PHQ9_Total",
    "GAD7_Total",
    "Burnout_Total",
    "Intent_Leave_Total",
    "Performance_Impact_Total",
    "Work_Withdrawal_Total",
]

CLASSIFICATION_OUTCOMES = ["PHQ9_ModeratePlus", "GAD7_ModeratePlus"]


def load_analysis_ready(csv_path: Path):
    raw_info = inspect_raw_csv(csv_path)
    raw_df = load_data(csv_path, raw_info)
    scored, _, _ = add_scores(raw_df, raw_info)
    ready, fixes_applied, missing_value_actions = apply_analysis_fixes(scored)
    return ready, fixes_applied, missing_value_actions


def feature_columns_for(outcome, data):
    candidates = [c for c in DEMOGRAPHIC_FEATURES + SCALE_FEATURES if c in data.columns and c != outcome]
    if outcome == "PHQ9_Total":
        candidates = [c for c in candidates if c not in {"PHQ9_ModeratePlus"}]
    if outcome == "GAD7_Total":
        candidates = [c for c in candidates if c not in {"GAD7_ModeratePlus"}]
    if outcome == "PHQ9_ModeratePlus":
        candidates = [c for c in candidates if c not in {"PHQ9_Total"}]
    if outcome == "GAD7_ModeratePlus":
        candidates = [c for c in candidates if c not in {"GAD7_Total"}]
    return candidates


def make_preprocessor(x):
    numeric = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def regression_models(x, thorough=False):
    preprocessor = make_preprocessor(x)
    models = {
        "Ridge regression": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=3.0, random_state=42)),
            ]
        )
    }
    if thorough:
        models["Random forest regression"] = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def classification_models(x, thorough=False):
    preprocessor = make_preprocessor(x)
    models = {
        "Logistic regression": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        )
    }
    if thorough:
        models["Random forest classifier"] = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def prepare_xy(data, outcome, features):
    frame = data[features + [outcome]].copy()
    frame = frame.replace({"Missing in source": np.nan})
    frame = frame.dropna(subset=[outcome])
    for col in features:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].fillna(frame[col].median())
        else:
            frame[col] = frame[col].fillna("Not specified")
    return frame[features], frame[outcome]


def cv_regression(outcome, x, y, thorough=False):
    rows = []
    best_name = None
    best_score = -np.inf
    best_model = None
    if thorough:
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        cv_label = "5-fold repeated 5x"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_label = "5-fold"
    for name, model in regression_models(x, thorough=thorough).items():
        print(f"  Training {outcome}: {name} ({cv_label})", flush=True)
        result = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error",
            },
        )
        r2 = result["test_r2"]
        mae = -result["test_neg_mae"]
        rmse = -result["test_neg_rmse"]
        rows.append(
            {
                "Outcome": outcome,
                "Task": "Regression",
                "Model": name,
                "N": len(y),
                "CV folds": cv_label,
                "Mean CV R2": np.mean(r2),
                "SD CV R2": np.std(r2),
                "Mean CV MAE": np.mean(mae),
                "Mean CV RMSE": np.mean(rmse),
                "Selection metric": "Higher CV R2; lower MAE/RMSE as secondary",
            }
        )
        if np.mean(r2) > best_score:
            best_score = np.mean(r2)
            best_name = name
            best_model = clone(model)
    best_model.fit(x, y)
    return rows, best_name, best_model


def cv_classification(outcome, x, y, thorough=False):
    rows = []
    best_name = None
    best_score = -np.inf
    best_model = None
    y_binary = y.map({"Yes": 1, "No": 0}).astype(int)
    min_class = y_binary.value_counts().min()
    if min_class < 5:
        return rows, best_name, best_model

    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
    for name, model in classification_models(x, thorough=thorough).items():
        print(f"  Training {outcome}: {name} ({cv.n_splits}-fold stratified)", flush=True)
        scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
        result = cross_validate(model, x, y_binary, cv=cv, scoring=scoring)
        auc = result["test_roc_auc"]
        rows.append(
            {
                "Outcome": outcome,
                "Task": "Classification",
                "Model": name,
                "N": len(y_binary),
                "Positive class count": int(y_binary.sum()),
                "CV folds": f"{cv.n_splits}-fold stratified",
                "Mean CV AUC": np.mean(auc),
                "Mean CV Accuracy": np.mean(result["test_accuracy"]),
                "Mean CV F1": np.mean(result["test_f1"]),
                "Selection metric": "Higher CV AUC",
            }
        )
        if np.mean(auc) > best_score:
            best_score = np.mean(auc)
            best_name = name
            best_model = clone(model)
    best_model.fit(x, y_binary)
    return rows, best_name, best_model


def prediction_percentile(values):
    series = pd.Series(values)
    return series.rank(pct=True).mul(100).round(1).to_numpy()


def feature_importance(model, x, y, outcome, model_name, task, thorough=False):
    if task == "Classification":
        y_eval = y.map({"Yes": 1, "No": 0}).astype(int)
        scoring = "roc_auc"
    else:
        y_eval = y
        scoring = "neg_mean_absolute_error"
    result = permutation_importance(
        model,
        x,
        y_eval,
        scoring=scoring,
        n_repeats=25 if thorough else 8,
        random_state=42,
        n_jobs=-1,
    )
    rows = []
    for feature, mean, sd in zip(x.columns, result.importances_mean, result.importances_std):
        rows.append(
            {
                "Outcome": outcome,
                "Task": task,
                "Best model": model_name,
                "Feature": feature,
                "Permutation importance": mean,
                "Importance SD": sd,
            }
        )
    return pd.DataFrame(rows).sort_values(["Outcome", "Permutation importance"], ascending=[True, False])


def run_ml_models(data, thorough=False):
    performance_rows = []
    importance_tables = []
    prediction_frame = data[["Timestamp", "Age", "Gender", "Designation", "Specialization"]].copy()
    selected_rows = []

    for outcome in REGRESSION_OUTCOMES:
        print(f"Regression outcome: {outcome}", flush=True)
        features = feature_columns_for(outcome, data)
        x, y = prepare_xy(data, outcome, features)
        rows, best_name, best_model = cv_regression(outcome, x, y, thorough=thorough)
        performance_rows.extend(rows)
        selected_rows.append({"Outcome": outcome, "Task": "Regression", "Selected model": best_name})
        print(f"  Calculating feature importance for {outcome}", flush=True)
        importance_tables.append(feature_importance(best_model, x, y, outcome, best_name, "Regression", thorough=thorough))
        preds = best_model.predict(x)
        prediction_frame[f"Predicted_{outcome}"] = preds.round(3)
        prediction_frame[f"{outcome}_Risk_Percentile"] = prediction_percentile(preds)

    for outcome in CLASSIFICATION_OUTCOMES:
        print(f"Classification outcome: {outcome}", flush=True)
        features = feature_columns_for(outcome, data)
        x, y = prepare_xy(data, outcome, features)
        rows, best_name, best_model = cv_classification(outcome, x, y, thorough=thorough)
        if not rows:
            continue
        performance_rows.extend(rows)
        selected_rows.append({"Outcome": outcome, "Task": "Classification", "Selected model": best_name})
        print(f"  Calculating feature importance for {outcome}", flush=True)
        importance_tables.append(feature_importance(best_model, x, y, outcome, best_name, "Classification", thorough=thorough))
        probs = best_model.predict_proba(x)[:, 1]
        prediction_frame[f"Predicted_Probability_{outcome}_Yes"] = probs.round(3)
        prediction_frame[f"{outcome}_Risk_Percentile"] = prediction_percentile(probs)

    performance = pd.DataFrame(performance_rows).round(4)
    selected = pd.DataFrame(selected_rows)
    importance = pd.concat(importance_tables, ignore_index=True).round(5)
    return performance, selected, importance, prediction_frame


def build_clusters(data):
    cluster_features = [c for c in SCALE_FEATURES if c in data.columns]
    matrix = data[cluster_features].copy()
    for col in cluster_features:
        matrix[col] = matrix[col].fillna(matrix[col].median())
    scaled = StandardScaler().fit_transform(matrix)

    score_rows = []
    best_k = 2
    best_score = -1
    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        score_rows.append({"k": k, "Silhouette score": score})
        if score > best_score:
            best_score = score
            best_k = k

    final = KMeans(n_clusters=best_k, n_init=100, random_state=42)
    labels = final.fit_predict(scaled)
    labeled = data[["Age", "Gender", "Designation", "Specialization"] + cluster_features].copy()
    labeled["Cluster"] = labels + 1

    profile = labeled.groupby("Cluster")[cluster_features].mean().round(3)
    profile.insert(0, "N", labeled["Cluster"].value_counts().sort_index())
    profile = profile.reset_index()

    descriptors = []
    for _, row in profile.iterrows():
        numeric = row[cluster_features].astype(float)
        high = numeric.sort_values(ascending=False).head(3)
        low = numeric.sort_values().head(2)
        descriptors.append(
            {
                "Cluster": int(row["Cluster"]),
                "N": int(row["N"]),
                "Profile label": "; ".join([f"high {idx}" for idx in high.index[:2]]),
                "Highest traits": "; ".join(f"{idx}={val:.2f}" for idx, val in high.items()),
                "Lowest traits": "; ".join(f"{idx}={val:.2f}" for idx, val in low.items()),
            }
        )
    cluster_descriptions = pd.DataFrame(descriptors)
    cluster_scores = pd.DataFrame(score_rows).round(4)
    return cluster_scores, profile, cluster_descriptions, labeled


def high_risk_segments(data):
    rows = []
    outcomes = ["Burnout_Total", "Intent_Leave_Total", "PHQ9_Total", "GAD7_Total", "Performance_Impact_Total"]
    for group_col in ["Gender", "Designation", "Specialization"]:
        for outcome in outcomes:
            overall = data[outcome].mean()
            for level, group in data.groupby(group_col):
                if len(group) < 3:
                    continue
                rows.append(
                    {
                        "Segment variable": group_col,
                        "Segment": level,
                        "Outcome": outcome,
                        "N": len(group),
                        "Segment mean": group[outcome].mean(),
                        "Overall mean": overall,
                        "Difference vs overall": group[outcome].mean() - overall,
                        "Segment median": group[outcome].median(),
                    }
                )
    return pd.DataFrame(rows).sort_values("Difference vs overall", ascending=False).round(4)


def make_insights(performance, importance, segments, cluster_descriptions, data):
    rows = []

    def add(category, insight, evidence, action):
        rows.append({"Category": category, "Insight": insight, "Evidence": evidence, "Suggested action": action})

    for outcome in REGRESSION_OUTCOMES:
        top = importance[importance["Outcome"] == outcome].head(3)
        perf = performance[(performance["Outcome"] == outcome) & (performance["Task"] == "Regression")].sort_values("Mean CV R2", ascending=False).head(1)
        if top.empty or perf.empty:
            continue
        top_text = "; ".join(f"{r.Feature} ({r['Permutation importance']:.3f})" for _, r in top.iterrows())
        add(
            "Model driver",
            f"Top modeled drivers for {outcome}",
            f"Best CV model: {perf.iloc[0]['Model']}; CV R2={perf.iloc[0]['Mean CV R2']:.3f}; top features: {top_text}",
            "Prioritize these variables when designing follow-up interviews or targeted support actions.",
        )

    top_segments = segments.head(8)
    for _, row in top_segments.iterrows():
        add(
            "High-risk segment",
            f"{row['Segment variable']}={row['Segment']} is elevated for {row['Outcome']}",
            f"N={int(row['N'])}; segment mean={row['Segment mean']:.2f}; overall mean={row['Overall mean']:.2f}; difference={row['Difference vs overall']:.2f}",
            "Review whether this segment needs tailored support, debriefing, or manager follow-up.",
        )

    for _, row in cluster_descriptions.iterrows():
        add(
            "Respondent profile",
            f"Cluster {int(row['Cluster'])}: {row['Profile label']}",
            f"N={int(row['N'])}; {row['Highest traits']}",
            "Use clusters to create interpretable respondent profiles rather than one-size-fits-all interventions.",
        )

    moderate_phq = (data["PHQ9_Total"] >= 10).mean() * 100
    moderate_gad = (data["GAD7_Total"] >= 10).mean() * 100
    add(
        "Clinical screen",
        "Moderate-or-higher symptom burden exists in a meaningful minority of responses",
        f"PHQ-9 >=10: {moderate_phq:.1f}%; GAD-7 >=10: {moderate_gad:.1f}%",
        "Interpret as screening-level signal only; clinical diagnosis requires appropriate assessment.",
    )
    return pd.DataFrame(rows)


def write_workbook(output_path, tables):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame({"Message": ["No rows generated"]})
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:200]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)


def main():
    parser = argparse.ArgumentParser(description="Generate ML insights from cleaned Second Victim survey data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to source CSV.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for Excel output.")
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="Run slower repeated CV, random forests, and more permutation repeats.",
    )
    args = parser.parse_args()

    mode = "thorough" if args.thorough else "fast"
    print(f"Running ML insights in {mode} mode.", flush=True)
    data, fixes_applied, missing_value_actions = load_analysis_ready(Path(args.input))
    performance, selected, importance, predictions = run_ml_models(data, thorough=args.thorough)
    print("Building respondent clusters", flush=True)
    cluster_scores, cluster_profile, cluster_descriptions, respondent_clusters = build_clusters(data)
    print("Building high-risk segment summaries", flush=True)
    segments = high_risk_segments(data)
    insights = make_insights(performance, importance, segments, cluster_descriptions, data)

    output_path = Path(args.output_dir) / OUTPUT_FILE
    tables = {
        "Executive_Insights": insights,
        "Model_Performance": performance,
        "Selected_Models": selected,
        "Top_Feature_Importance": importance,
        "High_Risk_Segments": segments,
        "Cluster_Selection": cluster_scores,
        "Cluster_Profiles": cluster_profile,
        "Cluster_Descriptions": cluster_descriptions,
        "Predicted_Risk_Scores": predictions,
        "Respondent_Clusters": respondent_clusters,
        "Fixes_Applied": fixes_applied,
        "Missing_Value_Actions": missing_value_actions,
    }
    write_workbook(output_path, tables)

    print("ML insights workbook created.")
    print(f"Rows analyzed: {len(data)}")
    print(f"Output workbook: {output_path}")
    print("\nTop insights:")
    print(insights.head(12).to_string(index=False))
    print(
        "\nNote: These ML models are exploratory because the dataset has 99 rows. "
        "Use cross-validated performance and feature importance as directional evidence, not causal proof."
    )


if __name__ == "__main__":
    main()
