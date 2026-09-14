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
    """Common save/close boilerplate. Was duplicated 4x across plotting funcs."""
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

DATA_PATH = Path(__file__).resolve().parent / "inputs" /"Dataset_finale.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"[load] shape={df.shape}")
    return df


# ---------------------------------------------------------------- 2. CLEAN (structural, pre-split)
def clean_structural(df):
    df = df.copy()
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
    return df


def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"[split] train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def clean_statistical(train_df, test_df, numeric_cols):
    """Fit outlier rule on TRAIN only, apply identical rule to both."""
    model = IsolationForest(contamination=0.02, random_state=42)
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


# ---------------------------------------------------------------- 3. ANALYZE
def analyze(df, label="train"):
    print(f"\n[analyze:{label}] describe:")
    print(df.describe())
    print(f"\n[analyze:{label}] domain_category counts:")
    print(df["domain_category"].value_counts())


# ---------------------------------------------------------------- 4. FEATURE ENGINEERING
def encode_time_cyclical(df):
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
    train_df = encode_time_cyclical(train_df.copy())
    test_df = encode_time_cyclical(test_df.copy())

    # one-hot encode domain_category, fit categories on train, align test
    train_df = pd.get_dummies(train_df, columns=["domain_category"], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=["domain_category"], drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(f"[feat] train cols: {list(train_df.columns)}")
    return train_df, test_df


def scale_features(train_df, test_df, feature_cols):
    """Fit scaler on train only, transform both."""
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled, scaler


# ---------------------------------------------------------------- 5. BASELINE / MODEL
def train_models(X_train, y_train):
    models = {}

    class MeanBaseline:
        def fit(self, X, y):
            self.value = y.mean()
            return self
        def predict(self, X):
            return np.full(len(X), self.value)

    baseline = MeanBaseline().fit(X_train, y_train)
    models["baseline_mean"] = baseline

    lr = LinearRegression().fit(X_train, y_train)
    models["linear_regression"] = lr

    dt = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_train, y_train)
    models["decision_tree"] = dt

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(X_train, y_train)
    models["random_forest"] = rf

    print(f"[model] trained: {list(models.keys())}")
    return models


# ---------------------------------------------------------------- 6. EVALUATE
def evaluate(models, X_test, y_test):
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
    result = pd.DataFrame(rows)
    print("\n[evaluate]")
    print(result.to_string(index=False))
    return result, preds


# ---------------------------------------------------------------- 6b. CORRELATION
def compute_correlations(df, target="response_time_ms"):
    """Pearson correlation between every numeric feature and the target.
    Saved once, on the (feature-engineered) TRAIN split only."""
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
    """Horizontal bar chart of Pearson r per feature, sorted by magnitude."""
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


# ---------------------------------------------------------------- 6c. NETWORKING-ANGLE BREAKDOWN
def analyze_network_components(df, target="response_time_ms"):
    """Split response time into measured network stages (DNS+TCP+TLS+RTT)
    vs. everything else (server processing, queueing, payload transfer).
    This is the data needed to interpret results 'from a networking
    perspective' rather than just reporting model scores."""
    comp = df.copy()
    comp["network_ms"] = (comp["dns_time_ms"] + comp["tcp_connect_ms"]
                           + comp["tls_time_ms"] + comp["rtt_ms"])
    comp["other_ms"] = comp[target] - comp["network_ms"]
    comp["network_pct"] = comp["network_ms"] / comp[target] * 100
    comp["other_pct"] = 100 - comp["network_pct"]

    by_domain = (comp.groupby("domain_category")[["network_ms", "other_ms",
                                                     "network_pct", "other_pct"]]
                 .mean().reset_index().sort_values("network_pct", ascending=False))

    out_path = OUT_DIR / "network_component_breakdown.csv"
    by_domain.to_csv(out_path, index=False)
    print(f"[net-breakdown] saved network vs. other time by domain -> {out_path}")

    # stacked bar chart: avg ms spent in network stages vs. "other"
    plt.figure(figsize=(7, max(4, 0.4 * len(by_domain))))
    y_pos = np.arange(len(by_domain))
    plt.barh(y_pos, by_domain["network_ms"], color="#4C72B0",
              label="Network time (DNS+TCP+TLS+RTT)")
    plt.barh(y_pos, by_domain["other_ms"], left=by_domain["network_ms"],
              color="#DD8452", label="Other time (server, queue, payload)")
    plt.yticks(y_pos, [nice(d) for d in by_domain["domain_category"]])
    plt.xlabel("Average Response Time (ms)")
    plt.title("Response Time Breakdown by Domain Category")
    plt.legend(loc="lower right", fontsize=8, frameon=False)
    plt.gca().invert_yaxis()

    fig_path = OUT_DIR / "network_component_breakdown.png"
    _save_fig(fig_path)
    print(f"[net-breakdown] saved breakdown diagram -> {fig_path}")
    return by_domain


# ---------------------------------------------------------------- 6d. ERROR ANALYSIS
def analyze_errors(test_raw, y_test, preds, model_name="random_forest", top_n=15):
    """Pull the worst-predicted test requests (largest |residual|) with their
    original, human-readable feature values, plus mean error by domain
    category. Feeds directly into the Chapter 5 'error analysis' section:
    which cases does the model get wrong, and is there a pattern."""
    pred = preds[model_name]
    residual = y_test.values - pred

    err_df = test_raw.loc[y_test.index].copy()
    err_df["actual_response_time_ms"] = y_test.values
    err_df["predicted_response_time_ms"] = pred
    err_df["residual_ms"] = residual
    err_df["abs_residual_ms"] = np.abs(residual)
    err_df = err_df.sort_values("abs_residual_ms", ascending=False)

    worst_path = OUT_DIR / f"error_analysis_{model_name}_top{top_n}.csv"
    err_df.head(top_n).to_csv(worst_path, index=False)
    print(f"[error] saved top {top_n} worst predictions ({model_name}) -> {worst_path}")

    by_domain = (err_df.groupby("domain_category")["abs_residual_ms"]
                 .agg(mean_abs_error="mean", count="count")
                 .reset_index().sort_values("mean_abs_error", ascending=False))
    domain_path = OUT_DIR / f"error_by_domain_{model_name}.csv"
    by_domain.to_csv(domain_path, index=False)
    print(f"[error] saved mean error by domain ({model_name}) -> {domain_path}")

    return err_df, by_domain


# ---------------------------------------------------------------- 7. VISUALIZE
def visualize_feature_target(df, target="response_time_ms"):
    """Feature-vs-target scatter plots. Property of the data, not any
    one model's output - generated once, independent of models."""
    feat_dir = OUT_DIR / "feature_vs_target"
    feat_dir.mkdir(exist_ok=True)

    numeric_feats = [c for c in df.select_dtypes(include="number").columns
                      if c != target]

    for feat in numeric_feats:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[feat], df[target], alpha=0.4, color="#4C72B0",
                    label="data point (one request)")
        plt.xlabel(nice(feat))
        plt.ylabel(nice(target))
        plt.title(f"{nice(feat)} vs {nice(target)}")
        plt.legend(loc="upper right", fontsize=8)
        _save_fig(feat_dir / f"{feat}_vs_target.png")

    print(f"[visualize] saved {len(numeric_feats)} feature-vs-target plots -> {feat_dir}")


def visualize(y_test, preds, results, model_names=None):
    if model_names is None:
        model_names = list(preds.keys())

    for model_name in model_names:
        pred = preds[model_name]
        residual = y_test.values - pred
        row = results[results["model"] == model_name].iloc[0]
        metrics_text = (f"MAE = {row['MAE']:.1f}\nRMSE = {row['RMSE']:.1f}\n"
                         f"R² = {row['R2']:.3f}\nAccuracy = {row['Accuracy_%']:.1f}%")
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
        plt.title(f"Actual vs Predicted — {title_model}")
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
        plt.ylabel("Residual: Actual − Predicted (ms)")
        plt.title(f"Residual Plot — {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        _save_fig(model_dir / "residual_plot.png")

        # 3. residual histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=30, color="#4C72B0",
                  label="number of test requests")
        plt.axvline(0, color="#C44E52", linestyle="--", label="zero error")
        plt.xlabel("Residual: Actual − Predicted (ms)")
        plt.ylabel("Number of Test Requests")
        plt.title(f"Residual Distribution — {title_model}")
        plt.legend(loc="upper right", fontsize=8, frameon=False)
        _save_fig(model_dir / "residual_hist.png")

        print(f"[visualize] saved plots for {model_name} -> {model_dir}")


# ---------------------------------------------------------------- 8. COEFFICIENTS
def extract_coefficients(models, feature_cols):
    """Per-feature weight for every model that has one:
    - LinearRegression -> coef_
    - DecisionTree / RandomForest -> feature_importances_
    - MeanBaseline -> has neither, so it's skipped."""
    rows = []
    for name, m in models.items():
        if hasattr(m, "coef_"):
            for feat, val in zip(feature_cols, m.coef_):
                rows.append({"model": name, "feature": feat,
                             "value": val, "value_type": "coefficient"})
        elif hasattr(m, "feature_importances_"):
            for feat, val in zip(feature_cols, m.feature_importances_):
                rows.append({"model": name, "feature": feat,
                             "value": val, "value_type": "importance"})
        else:
            print(f"[coef] {name} has no coefficients/importances, skipped")

    coef_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "model_coefficients.csv"
    coef_df.to_csv(out_path, index=False)
    print(f"[coef] saved model coefficients -> {out_path}")
    return coef_df


# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    df = load_data()
    df = clean_structural(df)
    train_df, test_df = split_data(df)

    numeric_cols = ["dns_time_ms", "tcp_connect_ms", "tls_time_ms", "rtt_ms",
                     "response_size", "response_time_ms"]
    train_df, test_df = clean_statistical(train_df, test_df, numeric_cols)

    analyze(train_df, "train")
    visualize_feature_target(train_df)
    analyze_network_components(train_df)

    test_raw = test_df.copy()  # human-readable, pre-encoding — used for error analysis later

    train_df, test_df = feature_engineer(train_df, test_df)

    feature_cols = [c for c in train_df.columns if c != "response_time_ms"]

    corr_df = compute_correlations(train_df)
    visualize_correlations(corr_df)

    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    X_train, y_train = train_df[feature_cols], train_df["response_time_ms"]
    X_test, y_test = test_df[feature_cols], test_df["response_time_ms"]

    models = train_models(X_train, y_train)
    results, preds = evaluate(models, X_test, y_test)
    visualize(y_test, preds, results)

    coef_df = extract_coefficients(models, feature_cols)

    best_model = results.loc[results["R2"].idxmax(), "model"]
    err_df, err_by_domain = analyze_errors(test_raw, y_test, preds, model_name=best_model)
    print(f"[error] worst-case analysis run on best model by R2: {best_model}")

    results.to_csv(OUT_DIR / "model_results.csv", index=False)
    print(f"\n[main] pipeline complete. results saved to {OUT_DIR/'model_results.csv'}")