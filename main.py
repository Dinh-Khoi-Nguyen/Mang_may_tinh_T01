import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

DISPLAY_NAME = {
    "dns_time_ms": "DNS Time (ms)",
    "tcp_connect_ms": "TCP Connect Time (ms)",
    "tls_time_ms": "TLS Handshake Time (ms)",
    "rtt_ms": "Round-Trip Time (ms)",
    "response_size": "Response Size (bytes)",
    "response_time_ms": "Response Time (ms)",
    "time_of_day": "Time of Day",
    "time_sin": "Time of Day (sin)",
    "time_cos": "Time of Day (cos)",
    "domain_category": "Domain Category",
    "baseline_mean": "Baseline (Mean)",
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def nice(name):
    """Human-readable label, falls back to title-cased underscores-to-spaces."""
    return DISPLAY_NAME.get(name, name.replace("_", " ").title())


def _save_fig(path):
    """Common save/close boilerplate."""
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


DATA_PATH = Path(__file__).resolve().parent / "inputs" / "Dataset_finale.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ==================================================================
# 1. LOAD
# ==================================================================
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"[load] shape={df.shape}")
    return df


# ==================================================================
# 2. CLEAN
#    2a. structural (null/dup, http_status, cross-field check) - pre-split
#    2b. split (train/test)
#    2c. statistical (Isolation Forest, fit on train only) - post-split
#    Nulls/duplicates are real, not hypothetical: Crawl_code_new.py appends
#    to Dataset_finale.csv across separate crawl runs (mode="a"), so
#    re-running the crawler can reintroduce duplicate rows; an interrupted
#    run can also leave partial/null rows. Both are dropped here.
# ==================================================================
def clean(df, numeric_cols, test_size=0.2, random_state=42, contamination=0.02):
    df = df.copy()

    # --- 2a. structural (pre-split) ---
    n_null, n_dup = df.isnull().sum().sum(), df.duplicated().sum()
    print(f"[clean] nulls={n_null}, dupes={n_dup}")
    if n_null:
        df = df.dropna()
    if n_dup:
        df = df.drop_duplicates()

    before = len(df)
    df = df[df["http_status"] == 200].copy()
    print(f"[clean] dropped {before - len(df)} non-200 rows")
    df = df.drop(columns=["http_status"])

    before = len(df)
    comp_sum = df["dns_time_ms"] + df["tcp_connect_ms"] + df["tls_time_ms"]
    df = df[comp_sum <= df["response_time_ms"]].copy()
    print(f"[clean] dropped {before - len(df)} cross-field-broken rows")

    df["domain_category"] = df["domain_category"].str.strip().str.lower()
    print(f"[clean] structural clean done, shape={df.shape}")

    # --- 2b. split ---
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[split] train={train_df.shape}, test={test_df.shape}")

    # --- 2c. statistical (Isolation Forest, fit on TRAIN only) ---
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(train_df[numeric_cols])

    train_pred = model.predict(train_df[numeric_cols])
    test_pred = model.predict(test_df[numeric_cols])

    train_clean = train_df[train_pred == 1].copy()
    test_clean = test_df[test_pred == 1].copy()

    print(f"[clean-stat] train: dropped {len(train_df)-len(train_clean)} anomalies "
          f"({len(train_clean)} remain)")
    print(f"[clean-stat] test: dropped {len(test_df)-len(test_clean)} anomalies "
          f"({len(test_clean)} remain)")

    return train_clean, test_clean


# ==================================================================
# 3. ANALYZE
# ==================================================================
def analyze(df, label="train"):
    print(f"\n[analyze:{label}] describe:")
    print(df.describe())
    df.describe().to_csv(OUT_DIR / f"describe_{label}.csv")
    print(f"\n[analyze:{label}] domain_category counts:")
    print(df["domain_category"].value_counts())
    df["domain_category"].value_counts().to_csv(OUT_DIR / f"domain_counts_{label}.csv")


# ==================================================================
# PEARSON CORRELATION — outside the Load->Clean->...->Visualize pipeline.
# Property of the (feature-engineered) data itself, independent of any
# model. Fit/computed on TRAIN only, never touches test.
# ==================================================================
def compute_correlations(df, target="response_time_ms"):
    numeric_feats = [c for c in df.select_dtypes(include=["number", "bool"]).columns
                      if c != target]

    rows = []
    for feat in numeric_feats:
        r = df[feat].astype(float).corr(df[target], method="pearson")
        rows.append({"feature": feat, "pearson_r": r})

    corr_df = pd.DataFrame(rows)
    corr_df["abs_r"] = corr_df["pearson_r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False).drop(columns="abs_r")

    out_path = OUT_DIR / "pearson_correlations.csv"
    corr_df.to_csv(out_path, index=False)
    print(f"[correlate] saved pearson correlations -> {out_path}")
    return corr_df


def visualize_correlations(corr_df):
    plt.figure(figsize=(7, max(4, 0.35 * len(corr_df))))
    labels = [nice(f) for f in corr_df["feature"]]
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in corr_df["pearson_r"]]

    plt.barh(labels, corr_df["pearson_r"], color=colors,
              label="Pearson r (feature vs. response time)")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.title("Feature Correlation with Response Time")
    plt.legend(loc="lower right", fontsize=8, frameon=False)
    plt.gca().invert_yaxis()

    out_path = OUT_DIR / "pearson_correlations.png"
    _save_fig(out_path)
    print(f"[correlate] saved correlation diagram -> {out_path}")


# ==================================================================
# 4. FEATURE ENGINEERING
#    cyclical time encoding + one-hot domain_category + scaling.
#    Categories/scaler fit on TRAIN only, applied to test.
# ==================================================================
def _encode_time_cyclical(df):
    td = df["time_of_day"]
    h, m, s = td // 10000, (td // 100) % 100, td % 100
    bad = (h > 23) | (m > 59) | (s > 59)
    if bad.any():
        raise ValueError(f"invalid HHMMSS values: {td[bad].tolist()}")
    total_seconds = h * 3600 + m * 60 + s
    theta = 2 * np.pi * (total_seconds / 86400)
    df["time_sin"] = np.sin(theta)
    df["time_cos"] = np.cos(theta)
    return df.drop(columns=["time_of_day"])


def feature_engineer(train_df, test_df):
    # cyclical time encoding
    train_df = _encode_time_cyclical(train_df.copy())
    test_df = _encode_time_cyclical(test_df.copy())

    # one-hot encode domain_category, fit categories on train, align test
    train_df = pd.get_dummies(train_df, columns=["domain_category"], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=["domain_category"], drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(f"[feat] train cols: {list(train_df.columns)}")

    # scale (fit on train only, transform both)
    feature_cols = [c for c in train_df.columns if c != "response_time_ms"]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, feature_cols, scaler


# ==================================================================
# 5. TRAIN MODELS
# ==================================================================
def train_models(X_train, y_train):
    models = {}

    class MeanBaseline:
        def fit(self, X, y):
            self.value = y.mean()
            return self
        def predict(self, X):
            return np.full(len(X), self.value)

    models["baseline_mean"] = MeanBaseline().fit(X_train, y_train)
    models["linear_regression"] = LinearRegression().fit(X_train, y_train)
    models["decision_tree"] = DecisionTreeRegressor(
        max_depth=6, random_state=42).fit(X_train, y_train)
    models["random_forest"] = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42).fit(X_train, y_train)

    print(f"[model] trained: {list(models.keys())}")
    return models


# ==================================================================
# 6. EVALUATE
#    metrics (MAE/RMSE/R2/MAPE/Accuracy) + coefficients/importances
#    + worst-prediction error analysis (non-baseline models)
# ==================================================================
def evaluate(models, X_test, y_test, test_raw, feature_cols,
             error_top_n=10, error_models=("linear_regression", "decision_tree", "random_forest")):
    # --- metrics ---
    rows = []
    preds = {}
    for name, m in models.items():
        pred = m.predict(X_test)
        preds[name] = pred
        mape = np.mean(np.abs((y_test.values - pred) / y_test.values)) * 100
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "MAPE_%": mape,
            "Accuracy_%": 100 - mape,
        })
    results = pd.DataFrame(rows)
    print("\n[evaluate]")
    print(results.to_string(index=False))

    # --- coefficients / feature importances ---
    coef_rows = []
    for name, m in models.items():
        if hasattr(m, "coef_"):
            for feat, val in zip(feature_cols, m.coef_):
                coef_rows.append({"model": name, "feature": feat,
                                   "value": val, "value_type": "coefficient"})
        elif hasattr(m, "feature_importances_"):
            for feat, val in zip(feature_cols, m.feature_importances_):
                coef_rows.append({"model": name, "feature": feat,
                                   "value": val, "value_type": "importance"})
        else:
            print(f"[coef] {name} has no coefficients/importances, skipped")
    coef_df = pd.DataFrame(coef_rows)
    coef_path = OUT_DIR / "model_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"[coef] saved model coefficients -> {coef_path}")

    # --- worst-prediction error analysis (non-baseline models) ---
    error_dfs = {}
    for name in error_models:
        pred = preds[name]
        residual = y_test.values - pred

        err_df = test_raw.loc[y_test.index].copy()
        err_df["actual_response_time_ms"] = y_test.values
        err_df["predicted_response_time_ms"] = pred
        err_df["residual_ms"] = residual
        err_df["abs_residual_ms"] = np.abs(residual)
        err_df = err_df.sort_values("abs_residual_ms", ascending=False).head(error_top_n)

        err_path = OUT_DIR / f"error_analysis_{name}_top{error_top_n}.csv"
        err_df.to_csv(err_path, index=False)
        print(f"[error] saved top {error_top_n} worst predictions ({name}) -> {err_path}")
        error_dfs[name] = err_df

    return results, preds, coef_df, error_dfs


# ==================================================================
# 7. VISUALIZE
#    feature-vs-target scatter (raw train) + actual-vs-predicted,
#    residual plot, residual histogram (per model, on test)
# ==================================================================
def visualize(train_raw, y_test, preds, results, target="response_time_ms", model_names=None):
    # --- feature vs target scatter (raw, pre-feature-engineering train) ---
    feat_dir = OUT_DIR / "feature_vs_target"
    feat_dir.mkdir(exist_ok=True)

    numeric_feats = [c for c in train_raw.select_dtypes(include="number").columns
                      if c != target]
    for feat in numeric_feats:
        plt.figure(figsize=(6, 4))
        plt.scatter(train_raw[feat], train_raw[target], alpha=0.4, color="#4C72B0",
                    label="data point (one request)")
        plt.xlabel(nice(feat))
        plt.ylabel(nice(target))
        plt.title(f"{nice(feat)} vs {nice(target)}")
        plt.legend(loc="upper right", fontsize=8)
        _save_fig(feat_dir / f"{feat}_vs_target.png")
    print(f"[visualize] saved {len(numeric_feats)} feature-vs-target plots -> {feat_dir}")

    # --- per-model diagnostic plots (test) ---
    if model_names is None:
        model_names = list(preds.keys())

    for model_name in model_names:
        pred = preds[model_name]
        residual = y_test.values - pred
        row = results[results["model"] == model_name].iloc[0]
        metrics_text = (f"MAE = {row['MAE']:.1f}\nRMSE = {row['RMSE']:.1f}\n"
                         f"R\u00b2 = {row['R2']:.3f}\nAccuracy = {row['Accuracy_%']:.1f}%")
        title_model = nice(model_name)

        model_dir = OUT_DIR / model_name
        model_dir.mkdir(exist_ok=True)

        # 1. actual vs predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, pred, alpha=0.5, color="#4C72B0",
                    label="test request (actual vs. predicted)")
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        plt.plot(lims, lims, color="#C44E52", linestyle="--",
                  label="perfect prediction line")
        plt.xlabel("Actual Response Time (ms)")
        plt.ylabel("Predicted Response Time (ms)")
        plt.title(f"Actual vs Predicted \u2014 {title_model}")
        plt.legend(loc="upper left", fontsize=8, frameon=False)
        plt.gca().text(0.98, 0.02, metrics_text, transform=plt.gca().transAxes,
                        fontsize=9, va="bottom", ha="right",
                        bbox=dict(boxstyle="round", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.9))
        _save_fig(model_dir / "actual_vs_predicted.png")

        # 2. residual plot
        plt.figure(figsize=(6, 4))
        plt.scatter(pred, residual, alpha=0.5, color="#4C72B0",
                    label="test request (error at that prediction)")
        plt.axhline(0, color="#C44E52", linestyle="--", label="zero error")
        plt.xlabel("Predicted Response Time (ms)")
        plt.ylabel("Residual: Actual \u2212 Predicted (ms)")
        plt.title(f"Residual Plot \u2014 {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        _save_fig(model_dir / "residual_plot.png")

        # 3. residual histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=30, color="#4C72B0",
                  label="number of test requests")
        plt.axvline(0, color="#C44E52", linestyle="--", label="zero error")
        plt.xlabel("Residual: Actual \u2212 Predicted (ms)")
        plt.ylabel("Number of Test Requests")
        plt.title(f"Residual Distribution \u2014 {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        _save_fig(model_dir / "residual_hist.png")

        print(f"[visualize] saved plots for {model_name} -> {model_dir}")


# ==================================================================
# MAIN — Load -> Clean -> Analyze -> Feature Engineering -> Train
#         -> Evaluate -> Visualize  (Pearson runs alongside, outside
#         the numbered pipeline, between Feature Engineering and Train)
# ==================================================================
if __name__ == "__main__":
    # 1. LOAD
    df = load_data()

    # 2. CLEAN
    numeric_cols = ["dns_time_ms", "tcp_connect_ms", "tls_time_ms", "rtt_ms",
                     "response_size", "response_time_ms"]
    train_df, test_df = clean(df, numeric_cols)

    # 3. ANALYZE
    analyze(train_df, "train")
    train_raw = train_df.copy()  # pre-feature-engineering, for feature-vs-target plots
    test_raw = test_df.copy()    # pre-encoding, human-readable, for error analysis

    # 4. FEATURE ENGINEERING
    train_df, test_df, feature_cols, scaler = feature_engineer(train_df, test_df)

    # PEARSON (outside the pipeline, train-only, feature-engineered data)
    corr_df = compute_correlations(train_df)
    visualize_correlations(corr_df)

    X_train, y_train = train_df[feature_cols], train_df["response_time_ms"]
    X_test, y_test = test_df[feature_cols], test_df["response_time_ms"]

    # 5. TRAIN
    models = train_models(X_train, y_train)

    # 6. EVALUATE
    results, preds, coef_df, error_dfs = evaluate(models, X_test, y_test, test_raw, feature_cols)

    # 7. VISUALIZE
    visualize(train_raw, y_test, preds, results)

    results.to_csv(OUT_DIR / "model_results.csv", index=False)
    print(f"\n[main] pipeline complete. results saved to {OUT_DIR/'model_results.csv'}")