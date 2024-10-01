import numpy as np
import numpy.typing as npt
import pandas as pd
import plotnine as pn
from typing import Any, Callable

def nexcp_split(
    model: Callable[[npt.NDArray, pd.DataFrame, npt.NDArray], Any],
    split_function: Callable[[int], npt.NDArray],
    y: npt.NDArray,
    X: pd.DataFrame,
    tag_function: Callable[[int], npt.NDArray],
    weight_function: Callable[[int], npt.NDArray],
    alpha: float,
    test_index: int,
    model_type: str = "sklearn",
    interval_type: str = "conformal",
    include_tags: bool = True,
    predict_proba: bool = False,
):
    """Implements non-exchangeable split conformal prediction"""
    split_indices = split_function(np.min(test_index))
    tags = tag_function(X.iloc[:np.min(test_index)])
    weights = weight_function(X.iloc[:np.min(test_index)])
    # Pull test observation(s) from data
    y_test = y[test_index]
    X_test = (
        X.iloc[test_index].to_frame().T 
        if isinstance(test_index, (int, np.integer)) 
        else X.iloc[test_index]
    )
    # Select all observations up to that point
    y = y[:np.min(test_index)]
    X = X.iloc[:np.min(test_index)]
    # Split data, tags, and weights
    X_train = X.iloc[split_indices]
    y_train = y[split_indices]
    X_calib = X.drop(split_indices)
    y_calib = np.delete(y, split_indices)
    # Train model
    if model_type == "sm":
        if include_tags:
            model = model(y_train, X_train.drop(labels="date", axis=1).astype(float), weights=tags[split_indices])
        else:
            model = model(y_train, X_train.drop(labels="date", axis=1).astype(float))
        model = model.fit()
    elif model_type == "sklearn":
        if include_tags:
            model.fit(X_train.drop(labels="date", axis=1), y_train, sample_weight=tags[split_indices])
        else:
            model.fit(X_train.drop(labels="date", axis=1), y_train)
    # Generate residuals
    if model_type == "sm":
        residuals = np.abs(y_calib - model.predict(X_calib.drop(labels="date", axis=1).astype(float)))
    elif model_type == "sklearn":
        if predict_proba:
            residuals = np.abs(y_calib - model.predict_proba(X_calib.drop(labels="date", axis=1))[:, 1])
        else:
            residuals = np.abs(y_calib - model.predict(X_calib.drop(labels="date", axis=1)))
    # Calculate weighted quantile of residuals
    weights_calib = normalize_weights(np.delete(weights[:np.min(test_index)], split_indices))
    q_hat = np.quantile(
        residuals,
        1 - alpha,
        weights=weights_calib,
        method="inverted_cdf"
    )
    # Calculate predicted value
    if model_type == "sm":
        y_hat = model.predict(X_test.drop(labels="date", axis=1).astype(float))
    elif model_type == "sklearn":
        y_hat = model.predict(X_test.drop(labels="date", axis=1))
    # Generate CI
    if interval_type == "normal":
        pred_obj = model.get_prediction(X_test.drop(labels="date", axis=1).astype(float))
        bounds = pred_obj.summary_frame(alpha=alpha)
        lb = bounds["obs_ci_lower"].to_numpy()
        ub = bounds["obs_ci_upper"].to_numpy()
    elif interval_type == "conformal":
        lb = y_hat - q_hat
        ub = y_hat + q_hat
    covered = (lb <= y_test) & (y_test <= ub)
    return {"ci": np.transpose(np.vstack([lb, y_hat, ub])), "covered": covered, "width": ub-lb}

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights / np.sum(weights)

def plot(
    results: pd.DataFrame,
    column: str,
    window: int = 10
):
    """Plot the algorithm's mean coverage over a sliding window"""

    results = (
        results
        .groupby([column, "model", "alpha"])[["covered", "width"]]
        .mean()
        .reset_index(drop=False)
    )
    results[["rolling_coverage", "rolling_width"]] = (
        results.groupby(["model", "alpha"])[["covered", "width"]]
        .transform(lambda x: x.rolling(window=window).mean())
    )
    results = results.groupby(column).filter(lambda x: x["rolling_coverage"].notna().all() and x["rolling_width"].notna().all())
    results = results.melt(
        id_vars=[column, "model", "alpha", "covered", "width"], 
        value_vars=["rolling_coverage", "rolling_width"], 
        var_name="Metric",
        value_name="value"
    )
    results["Confidence level"] = 1-results["alpha"]
    results.loc[results["Metric"] == "rolling_width", "alpha"] = np.nan
    results["Metric"] = results["Metric"].apply(lambda x: "Interval coverage" if x == "rolling_coverage" else "Interval width")
    results["Method"] = results["model"].replace({
        "conf": "SCP",
        "weighted_conf": "WCP",
        "standard": "OLS"
    })
    coverage_plot = (
        pn.ggplot(
            results,
            pn.aes(x=column, y="value", color="Method", group="Method")
        )
        + pn.geom_line()
        + pn.facet_wrap(("Metric", "Confidence level"), nrow=2, ncol=2, scales="free",
                        labeller="label_context")
        + pn.geom_hline(
            data=results,
            mapping=pn.aes(yintercept="1-alpha"),
            linetype="dashed",
            inherit_aes=False,
            color="black"
        )
        + pn.theme_538()
        + pn.labs(x=column.title(), y="", color="", title="Predicting pitch-level hitting results")
        + pn.theme(
            legend_position="bottom",
            plot_title=pn.element_text(weight="bold", hjust=0.5)
        )
    )
    return coverage_plot

def total_bases(result):
    if result == "single":
        return 1
    elif result == "double":
        return 2
    elif result == "triple":
        return 3
    elif result == "home_run":
        return 4
    else:
        return 0