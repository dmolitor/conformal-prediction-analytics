from great_tables import GT, md
from pybaseball import statcast_batter, playerid_lookup
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from tqdm import tqdm
from utils import *
import warnings

warnings.filterwarnings(action="ignore")
generator = np.random.default_rng(123)
np.random.seed(111)

if __name__ == "__main__":

    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    analysis_cols = [
        "game_date", "events", "pitch_type", "release_speed", "release_spin_rate",
        "release_pos_x", "release_pos_y", "release_pos_z", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
        "effective_speed", "release_extension", "pitch_number", "zone", "p_throws",
        "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
        "at_bat_number", "on_1b", "on_2b", "on_3b"
    ]

    batter_id = playerid_lookup("ohtani", "shohei")["key_mlbam"].iloc[0]

    batter_pitches = (
        statcast_batter(start_dt="2024-03-28", end_dt="2024-09-30", player_id=batter_id)
        [analysis_cols]
        .assign(
            date=lambda df: pd.to_datetime(df["game_date"]),
            events=lambda df: df["events"].apply(total_bases),
            runner_on_1b=lambda df: df["on_1b"].notna().astype(int),
            runner_on_2b=lambda df: df["on_2b"].notna().astype(int),
            runner_on_3b=lambda df: df["on_3b"].notna().astype(int)
        )
        .drop(columns=["on_1b", "on_2b", "on_3b", "game_date"])
        .astype({
            col: "category" for col in ["pitch_type", "p_throws", "inning_topbot", "zone"]
        })
        .pipe(lambda df: pd.get_dummies(
            df,
            columns=["pitch_type", "p_throws", "inning_topbot", "zone"],
            drop_first=True
        ))
        .sort_values(by=["date", "at_bat_number", "pitch_number"])
        .dropna()
        .reset_index(drop=True)
    )

    pitches = np.empty(0)
    covered = np.empty(0)
    width = np.empty(0)
    models = np.empty(0)
    alphas = np.empty(0)

    for interval in ["weighted_conf", "conf", "standard"]:

        for pitch in tqdm(range(250, len(batter_pitches)), total=len(batter_pitches)-250):

            for alpha in [0.01, 0.1]:

                split_fn = lambda x: np.sort(generator.choice(x, int(np.floor(x*0.5)), replace=False))
                if interval in ["weighted_conf", "weighted_conf_reg"]:
                    def weight_fn(df):
                        return np.log(np.array([x + 1 for x in range(len(df))]))
                    def tag_fn(df):
                        return np.array([1.]*len(df))
                else:
                    def weight_fn(df):
                        return np.array([1.]*len(df))
                    tag_fn = weight_fn
                if interval == "standard":
                    interval_type = "normal"
                else:
                    interval_type = "conformal"
                test_index = pitch
                X = batter_pitches.drop(labels=["events"], axis=1)
                y = batter_pitches["events"].to_numpy()
                model = sm.WLS
                model_type = "sm"
                result = nexcp_split(
                    model=model,
                    split_function=split_fn,
                    y=y,
                    X=X,
                    tag_function=tag_fn,
                    weight_function=weight_fn,
                    alpha=alpha,
                    test_index=test_index,
                    model_type=model_type,
                    interval_type=interval_type
                )

                pitches = np.append(pitches, pitch)
                covered = np.append(covered, result["covered"])
                width = np.append(width, result["width"])
                models = np.append(models, interval)
                alphas = np.append(alphas, alpha)

    results_df = pd.DataFrame({
        "pitch": pitches,
        "covered": covered,
        "width": width,
        "model": models,
        "alpha": alphas
    })
    coverage_df = (
        results_df
        .groupby(["model", "alpha"])[["covered", "width"]]
        .mean()
        .reset_index(drop=False)
        .sort_values(by="alpha")
        .assign(
            covered = lambda df: df["covered"].apply(lambda x: round(x, 2)),
            width = lambda df: df["width"].apply(lambda x: round(x, 2))
        )
        .reset_index(drop=True)
    )
    
    ## Style the table with Great Tables
    coverage_df["model"] = coverage_df["model"].replace({
        "conf": "SCP",
        "weighted_conf": "WCP",
        "standard": "OLS"
    })

    # Pivot table with column spanners for confidence levels
    pivoted = coverage_df.pivot(index="model", columns="alpha", values=["covered", "width"])
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Create the table with bolded row names
    table = GT(pivoted, rowname_col="model") # Drop multi-index for GT compatibility
    table = table.tab_spanner(md("**Confidence level 90%**"), ["covered_0.1", "width_0.1"])
    table = table.tab_spanner(md("**Confidence level 99%**"), ["covered_0.01", "width_0.01"])
    table = table.cols_label({
        "covered_0.1": "Coverage",
        "width_0.1": "Width",
        "covered_0.01": "Coverage",
        "width_0.01": "Width"
    })
    table.save(file=str(fig_dir / "coverage_and_width_table.png"), scale=5)
    
    # Output plot with Plotnine
    pred_plot = plot(results_df, "pitch", window=500)
    pred_plot.save(filename=fig_dir / "coverage_and_width_graph.png", dpi=500, width=8, height=5)
