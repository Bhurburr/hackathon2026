import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

COL_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target",
]
BINARY_COLS        = ["sex", "fbs", "exang"]
SENTINEL_ZERO_COLS = ["trestbps", "chol", "thalach"]

FEATURE_DESCRIPTIONS = {
    "age":               "Age (years)",
    "sex":               "Sex (1=male 0=female)",
    "cp":                "Chest pain type",
    "trestbps":          "Resting blood pressure",
    "chol":              "Serum cholesterol",
    "fbs":               "Fasting blood sugar >120",
    "restecg":           "Resting ECG result",
    "thalach":           "Max heart rate",
    "exang":             "Exercise-induced angina",
    "oldpeak":           "ST depression",
    "slope":             "Slope of ST segment",
    "ca":                "Major vessels (fluoroscopy)",
    "thal":              "Thalassemia type",
    "ca_was_missing":    "CA value was missing",
    "thal_was_missing":  "Thal value was missing",
    "slope_was_missing": "Slope value was missing",
    "chol_was_missing":  "Chol value was missing",
}


# ============================================================
# PHASE 1 — LOAD AND SPLIT
# ============================================================

def load_processed(data_folder):
    files = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data",
    ]
    dfs = []
    print("Loading processed files:")
    for fname in files:
        path = os.path.join(data_folder, fname)
        if not os.path.exists(path):
            print(f"  {fname} NOT FOUND")
            continue
        df = pd.read_csv(
            path,
            header=None,
            names=COL_NAMES,
            na_values="?",
            sep=",",
            engine="python",
        )
        df = df.dropna(how="all")
        df["source"] = fname.replace(".data", "")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["target"] = (df["target"] > 0).astype(int)
        for col in SENTINEL_ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        dfs.append(df)
        print(f"  {fname:<40} {len(df):>4} rows")
    combined = pd.concat(dfs, ignore_index=True)
    rate = combined["target"].mean()
    print(f"\n  Total: {len(combined)} rows  disease rate: {rate:.1%}")
    return combined


def add_missingness_indicators(df):
    df   = df.copy()
    cols = [
        c for c in df.columns
        if c not in ["target", "source"]
        and df[c].isnull().any()
    ]
    for col in cols:
        df[col + "_was_missing"] = df[col].isnull().astype(int)
    print(f"  Missingness indicators added for: {cols}")
    return df, cols


def positional_split(df, train_frac=0.70, val_frac=0.15):
    n     = len(df)
    t     = int(n * train_frac)
    v     = int(n * (train_frac + val_frac))
    train = df.iloc[:t].copy()
    val   = df.iloc[t:v].copy()
    test  = df.iloc[v:].copy()
    tr = train["target"].mean()
    vr = val["target"].mean()
    ter = test["target"].mean()
    print(f"  Train: {len(train)} rows  outcome: {tr:.1%}")
    print(f"  Val:   {len(val)} rows  outcome: {vr:.1%}")
    print(f"  Test:  {len(test)} rows  outcome: {ter:.1%}")
    return train, val, test


def split_xy(df, drop_cols=None):
    drop_cols    = drop_cols or []
    cols_to_drop = [c for c in ["target"] + drop_cols if c in df.columns]
    return df.drop(columns=cols_to_drop), df["target"]


# ============================================================
# PHASE 2 — PREPROCESSING  (MICE + SMOTE)
# ============================================================

def build_preprocessor(numeric_cols, binary_cols):
    mice_pipe = Pipeline([
        ("mice", IterativeImputer(
            max_iter=10,
            random_state=42,
            min_value=0,
            imputation_order="ascending",
        )),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", IterativeImputer(
            max_iter=10,
            random_state=42,
            min_value=0,
            max_value=1,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", mice_pipe,   numeric_cols),
            ("binary",  binary_pipe, binary_cols),
        ],
        remainder="passthrough",
    )


def run_preprocessing(X_train, X_val, X_test, y_train,
                      numeric_cols, binary_cols, indicator_cols):
    print("  Fitting MICE imputer on training data only...")
    pre    = build_preprocessor(numeric_cols, binary_cols)

    X_tr_p = pre.fit_transform(X_train)
    X_v_p  = pre.transform(X_val)
    X_te_p = pre.transform(X_test)

    out_cols = numeric_cols + binary_cols + indicator_cols
    X_tr_df  = pd.DataFrame(X_tr_p, columns=out_cols)
    X_v_df   = pd.DataFrame(X_v_p,  columns=out_cols)
    X_te_df  = pd.DataFrame(X_te_p, columns=out_cols)

    assert X_tr_df.isnull().sum().sum() == 0, "NaNs remain in train"
    assert X_v_df.isnull().sum().sum()  == 0, "NaNs remain in val"
    assert X_te_df.isnull().sum().sum() == 0, "NaNs remain in test"
    print("  Imputation complete — no missing values remain.")

    before = pd.Series(y_train).value_counts().to_dict()
    print(f"  Before SMOTE: {before}")
    smote        = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_tr_df, y_train)
    after = pd.Series(y_bal).value_counts().to_dict()
    print(f"  After SMOTE:  {after}")

    return X_bal, y_bal, X_v_df, X_te_df, pre, out_cols


# ============================================================
# PHASE 3 — MODELING AND EVALUATION
# ============================================================

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
    fitted = {}
    print("  Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"    {name}: done")
    return fitted

def recalibrate_model(model, X_val, y_val, method="sigmoid"):
    """
    Applies Platt scaling (method='sigmoid') or isotonic
    regression (method='isotonic') to the selected model.

    Fitted on validation set only — never on training data.
    This corrects the probability scale so that when the
    model says 70% risk, roughly 70% of similar patients
    actually have the disease.

    method='sigmoid'  = Platt scaling — best for small datasets
    method='isotonic' = more flexible — needs larger datasets
    """
    print(f"\n  Recalibrating with Platt scaling ({method})...")

    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv="prefit"       # model already trained — just wrap it
    )
    calibrated.fit(X_val, y_val)

    # Verify improvement on validation set
    y_prob_before = model.predict_proba(X_val)[:, 1]
    y_prob_after  = calibrated.predict_proba(X_val)[:, 1]

    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    brier_before = brier_score_loss(y_val, y_prob_before)
    brier_after  = brier_score_loss(y_val, y_prob_after)

    # ECE before
    n_bins    = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    def compute_ece(y_true, y_prob):
        ece = 0.0
        for i in range(n_bins):
            mask = ((y_prob >= bin_edges[i]) &
                    (y_prob < bin_edges[i + 1]))
            if mask.sum() > 0:
                ece += mask.sum() * abs(
                    np.array(y_true)[mask].mean() -
                    y_prob[mask].mean())
        return ece / len(y_true)

    ece_before = compute_ece(y_val, y_prob_before)
    ece_after  = compute_ece(y_val, y_prob_after)

    print(f"\n  {'Metric':<10} {'Before':>10} {'After':>10} "
          f"{'Change':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'Brier':<10} {brier_before:>10.3f} "
          f"{brier_after:>10.3f} "
          f"{brier_after - brier_before:>+10.3f}")
    print(f"  {'ECE':<10} {ece_before:>10.3f} "
          f"{ece_after:>10.3f} "
          f"{ece_after - ece_before:>+10.3f}")

    ece_improvement = ece_before - ece_after
    brier_improvement = brier_before - brier_after

    if ece_improvement > 0 and brier_improvement > 0:
        print(f"\n  Platt scaling improved both ECE and Brier.")
        print(f"  ECE:   {ece_improvement:.3f} point reduction "
              f"({ece_improvement / ece_before:.0%} improvement)")
        print(f"  Brier: {brier_improvement:.3f} point reduction")
    elif ece_improvement > 0:
        print(f"\n  Platt scaling improved ECE by "
              f"{ece_improvement:.3f} points.")
    else:
        print(f"\n  Note: Platt scaling did not improve ECE on")
        print(f"  validation set. Model may already be well")
        print(f"  calibrated or dataset is too small for Platt.")
        print(f"  Using calibrated model for consistency.")

    return calibrated

def evaluate_all_models(models, X_val, y_val):
    from sklearn.metrics import brier_score_loss

    results = {}
    print("\n" + "="*50)
    print("  Validation set evaluation")
    print("="*50)

    n_bins    = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # ── Weighting scheme ─────────────────────────────────
    # AUROC  40% — discrimination is the foundation
    # ECE    30% — calibration purity matters for clinical
    #              probability communication
    # Brier  30% — combined penalty catches models that are
    #              good at ranking but terrible at probability
    #              estimates overall
    #
    # All three are needed because:
    # AUROC alone misses calibration problems (XGBoost case)
    # ECE alone misses discrimination problems
    # Brier alone can mask specific calibration failure modes
    W_AUROC  = 0.40
    W_ECE    = 0.30
    W_BRIER  = 0.30

    for name, model in models.items():
        y_prob = np.array(
            model.predict_proba(X_val)[:, 1]).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        y_val_arr = np.array(y_val).ravel()

        auroc  = float(roc_auc_score(y_val_arr, y_prob))
        pr_auc = float(average_precision_score(y_val_arr, y_prob))
        f1     = float(f1_score(y_val_arr, y_pred))
        brier  = float(brier_score_loss(y_val_arr, y_prob))

        # ECE
        ece = 0.0
        for i in range(n_bins):
            mask = ((y_prob >= bin_edges[i]) &
                    (y_prob < bin_edges[i + 1]))
            if mask.sum() > 0:
                ece += float(mask.sum()) * float(
                    np.mean(np.abs(
                        y_val_arr[mask] - y_prob[mask])))
        ece /= float(len(y_val_arr))

        # Combined score
        # ECE and Brier are inverted (lower = better → higher = better)
        combined = (W_AUROC  *  auroc +
                    W_ECE    * (1 - ece) +
                    W_BRIER  * (1 - brier))

        results[name] = {
            "model":    model,
            "y_prob":   y_prob,
            "auroc":    auroc,
            "pr_auc":   pr_auc,
            "f1":       f1,
            "brier":    brier,
            "ece":      ece,
            "combined": combined,
        }

        print(f"\n  {name}")
        print(f"    AUROC  (discrimination):   {auroc:.3f}  "
              f"weight={W_AUROC:.0%}")
        print(f"    ECE    (calibration):      {ece:.3f}  "
              f"weight={W_ECE:.0%}  lower=better")
        print(f"    Brier  (combined penalty): {brier:.3f}  "
              f"weight={W_BRIER:.0%}  lower=better")
        print(f"    PR-AUC:                    {pr_auc:.3f}")
        print(f"    F1:                        {f1:.3f}")
        print(f"    Combined score:            {combined:.3f}")

    # Select best model by combined score
    best = max(results, key=lambda k: results[k]["combined"])

    # Print comparison table
    print(f"\n  {'Model':<25} {'AUROC':>7} {'ECE':>7} "
          f"{'Brier':>7} {'Combined':>10}")
    print(f"  {'-'*60}")
    for name, res in results.items():
        marker = " <-- SELECTED" if name == best else ""
        print(f"  {name:<25} {res['auroc']:>7.3f} "
              f"{res['ece']:>7.3f} {res['brier']:>7.3f} "
              f"{res['combined']:>10.3f}{marker}")

    print(f"\n  Selection basis: combined score")
    print(f"  = {W_AUROC:.0%} x AUROC")
    print(f"  + {W_ECE:.0%} x (1 - ECE)")
    print(f"  + {W_BRIER:.0%} x (1 - Brier)")
    print(f"\n  Clinical rationale:")
    print(f"  AUROC alone would select XGBoost despite poor")
    print(f"  calibration. Including ECE and Brier ensures the")
    print(f"  selected model produces trustworthy probability")
    print(f"  estimates — critical when showing risk scores")
    print(f"  to clinicians who act on those numbers.")

    return results, best


def plot_evaluation_curves(results, y_val, best_name):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Model evaluation — heart disease prediction",
        fontsize=14, y=1.01,
    )
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors = {
        "Logistic Regression": "#5B8DB8",
        "Random Forest":       "#2E8B57",
        "XGBoost":             "#CC6633",
    }

    for name, res in results.items():
        yp  = res["y_prob"]
        col = colors[name]
        lw  = 2.5 if name == best_name else 1.2

        fpr, tpr, _    = roc_curve(y_val, yp)
        prec, rec, thr = precision_recall_curve(y_val, yp)
        f1s            = (2 * prec[:-1] * rec[:-1] /
                          (prec[:-1] + rec[:-1] + 1e-8))
        bi             = np.argmax(f1s)
        fp, mp         = calibration_curve(
            y_val, yp, n_bins=8, strategy="uniform")
        tr             = np.linspace(0.01, 0.99, 100)
        f1t            = [
            f1_score(y_val, (yp >= t).astype(int),
                     zero_division=0)
            for t in tr
        ]

        ax1.plot(fpr, tpr, color=col, lw=lw,
                 label=f"{name} ({res['auroc']:.3f})")
        ax2.plot(rec, prec, color=col, lw=lw,
                 label=f"{name} (AP={res['pr_auc']:.3f})")
        ax2.scatter(rec[bi], prec[bi],
                    color=col, s=60, zorder=5)
        ax3.plot(mp, fp, "s-", color=col, lw=lw, label=name)
        ax4.plot(tr, f1t, color=col, lw=lw,
                 label=f"{name} (best={thr[bi]:.2f})")
        ax4.axvline(thr[bi], color=col,
                    linestyle="--", alpha=0.4, lw=1)

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax1.set(xlabel="False positive rate",
            ylabel="True positive rate",
            title="AUROC  (primary metric)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set(xlabel="Recall", ylabel="Precision",
            title="Precision-recall  (threshold selection)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.plot([0, 1], [0, 1], "k--", lw=0.8,
             alpha=0.4, label="Perfect")
    ax3.set(xlabel="Mean predicted probability",
            ylabel="Fraction of positives",
            title="Calibration  (probability reliability)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4.set(xlabel="Decision threshold",
            ylabel="F1 score",
            title="F1 vs threshold  (choose cutoff)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    out = os.path.join(BASE_DIR, "evaluation_curves.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  Saved: {out}")


# ============================================================
# SHAP — INTERPRETABILITY
# ============================================================

def compute_shap(model, X_train, X_test,
                 feature_names, model_name):
    print(f"  Computing SHAP values for {model_name}...")
    if model_name in ["Random Forest", "XGBoost"]:
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X_test)
        if isinstance(vals, list):
            vals = vals[1]
    else:
        exp  = shap.LinearExplainer(model, X_train)
        vals = exp.shap_values(X_test)
    return exp, vals


def plot_shap_summary(shap_vals, X_test, feature_names):
    readable = [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
    X_disp   = pd.DataFrame(X_test, columns=readable)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_disp,
        feature_names=readable, show=False,
    )
    plt.title(
        "SHAP summary — feature impact on heart disease risk",
        fontsize=12, pad=15,
    )
    plt.tight_layout()
    out = os.path.join(BASE_DIR, "shap_summary.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  Saved: {out}")


def explain_single_patient(model, explainer, shap_vals,
                            X_test, y_test, feature_names,
                            patient_index=0):
    patient    = X_test.iloc[[patient_index]]
    y_prob     = model.predict_proba(patient)[0, 1]
    true_label = y_test.iloc[patient_index]
    p_shap     = shap_vals[patient_index]

    if y_prob >= 0.70:
        risk = "HIGH"
        rec  = "Recommend cardiology referral and further workup."
    elif y_prob >= 0.40:
        risk = "MODERATE"
        rec  = "Recommend lifestyle intervention and 3-month follow-up."
    else:
        risk = "LOW"
        rec  = "Recommend routine monitoring and preventive care."

    shap_df = pd.DataFrame({
        "feature":      feature_names,
        "shap_value":   p_shap,
        "actual_value": patient.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    top_risk    = shap_df[shap_df["shap_value"] > 0].head(3)
    top_protect = shap_df[shap_df["shap_value"] < 0].head(2)

    outcome_str = "Disease present" if true_label == 1 else "No disease"

    print("\n" + "=" * 55)
    print("  PATIENT RISK ASSESSMENT REPORT")
    print("=" * 55)
    print(f"  Patient:       #{patient_index}")
    print(f"  Risk score:    {y_prob:.1%}")
    print(f"  Risk level:    {risk}")
    print(f"  True outcome:  {outcome_str}")
    print(f"  Recommendation: {rec}")

    print("  Factors INCREASING risk:")
    for _, row in top_risk.iterrows():
        fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
        val   = row["actual_value"]
        imp   = row["shap_value"]
        print(f"  +  {fname:<35} value={val:.2f}  impact=+{imp:.3f}")

    print("  Factors DECREASING risk:")
    for _, row in top_protect.iterrows():
        fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
        val   = row["actual_value"]
        imp   = row["shap_value"]
        print(f"  -  {fname:<35} value={val:.2f}  impact={imp:.3f}")

    print("  Impact = how much each feature shifted the prediction")
    print("  from the average patient baseline.")
    print("  Positive = pushed risk higher. Negative = pushed risk lower.")
    print("  DISCLAIMER: This tool supports clinical decision-making")
    print("  and does not constitute a medical diagnosis.")
    print("=" * 55)

    base_val = (
        explainer.expected_value
        if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1]
    )
    shap_exp = shap.Explanation(
        values        = p_shap,
        base_values   = base_val,
        data          = patient.values[0],
        feature_names = [
            FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names
        ],
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, show=False)
    plt.title(
        f"Patient #{patient_index} — risk score {y_prob:.1%}",
        fontsize=11, pad=15,
    )
    plt.tight_layout()
    out = os.path.join(BASE_DIR, f"shap_patient_{patient_index}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"  Saved: {out}")
    return shap_df


def final_test_evaluation(model, name, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(
        y_test, y_pred,
        target_names=["No disease", "Disease"],
    )
    print("\n" + "=" * 50)
    print(f"  Final test evaluation — {name}")
    print("=" * 50)
    print(f"  AUROC:   {roc_auc_score(y_test, y_prob):.3f}")
    print(f"  PR-AUC:  {average_precision_score(y_test, y_prob):.3f}")
    print(f"  F1:      {f1_score(y_test, y_pred):.3f}")
    print(report)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("  PHASE 1 — Load and split")
    print("=" * 50)

    df = load_processed(DATA_FOLDER)
    df, missing_cols  = add_missingness_indicators(df)
    indicator_cols    = [c + "_was_missing" for c in missing_cols]

    print("\nSplitting data...")
    train, val, test  = positional_split(df)

    X_train, y_train  = split_xy(train, drop_cols=["source"])
    X_val,   y_val    = split_xy(val,   drop_cols=["source"])
    X_test,  y_test   = split_xy(test,  drop_cols=["source"])

    indicator_present = [c for c in indicator_cols
                         if c in X_train.columns]
    binary_present    = [c for c in BINARY_COLS
                         if c in X_train.columns]
    skip              = set(indicator_present + binary_present)
    numeric_cols      = [
        c for c in X_train.columns
        if c not in skip
        and X_train[c].dtype in ["float64", "int64"]
    ]

    print(f"  Numeric cols:   {numeric_cols}")
    print(f"  Binary cols:    {binary_present}")
    print(f"  Indicator cols: {indicator_present}")

    print("\n" + "=" * 50)
    print("  PHASE 2 — Preprocessing (MICE + SMOTE)")
    print("=" * 50)

    (X_tr_bal, y_tr_bal,
     X_v_df, X_te_df,
     preprocessor,
     feature_names) = run_preprocessing(
        X_train, X_val, X_test, y_train,
        numeric_cols, binary_present, indicator_present,
    )

    print("\n" + "=" * 50)
    print("  PHASE 3 — Modeling, evaluation, SHAP")
    print("=" * 50)

    models         = train_models(X_tr_bal, y_tr_bal)
    results, best  = evaluate_all_models(models, X_v_df, y_val)

    print("\nGenerating evaluation curves...")
    plot_evaluation_curves(results, y_val, best)

    best_model = models[best]

    # Recalibrate best model with Platt scaling
    # Fitted on validation set — never training or test
    print("\nApplying Platt scaling to best model...")
    best_model_calibrated = recalibrate_model(
        best_model, X_v_df, y_val, method="sigmoid"
    )

    # SHAP uses the original model (TreeExplainer needs raw model)
    # Calibration only wraps the probability output, not the
    # underlying tree structure that SHAP needs to analyze
    exp, shap_vals = compute_shap(
        best_model, X_tr_bal, X_te_df,
        feature_names, best
    )

    print("\nGenerating SHAP summary...")
    plot_shap_summary(shap_vals, X_te_df.values, feature_names)

    print("\nGenerating patient explanation...")
    explain_single_patient(
        best_model, exp, shap_vals,
        X_te_df, y_test, feature_names,
        patient_index=0,
    )

    final_test_evaluation(
        best_model_calibrated, best, X_te_df, y_test)

    # Save the calibrated model for the dashboard
    import joblib

    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    # Save background sample for SHAP KernelExplainer
    # Needs real training data distribution to compute
    # meaningful probability-scale SHAP values
    background_sample = X_tr_bal.sample(
        n=min(100, len(X_tr_bal)),
        random_state=42
    )

    joblib.dump({
        "model": best_model_calibrated,
        "model_raw": best_model,
        "model_name": best,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "binary_cols": binary_present,
        "numeric_cols": numeric_cols,
        "indicator_cols": indicator_present,
        "background": background_sample,
    }, model_path)
    print(f"\n  Model saved to: {model_path}")
    print("  Keys saved: model, model_raw, model_name,")
    print("              feature_names, preprocessor,")
    print("              binary_cols, numeric_cols,")
    print("              indicator_cols, background")
    print("  The dashboard will load this file.")

    print("\n" + "=" * 50)
    print("  Pipeline complete.")
    print("=" * 50)
    print(f"  Output files saved to: {BASE_DIR}")
    print("  - evaluation_curves.png")
    print("  - shap_summary.png")
    print("  - shap_patient_0.png")