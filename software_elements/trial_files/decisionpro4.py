import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ============================================================
# 1) Basic configuration
# ============================================================
EXCEL_FILE = "SP500_09_19_V2_SMA.xlsx"
SHEET_NAME = "MASTER"
RANDOM_STATE = 42

MACRO_FEATURES = [
    "VIX", "WTI (Oil)", "DGS10", "DGS2", "T10Y3M", "DFF",
    "CPIAUCSL", "UNRATE", "INDPRO", "T10Y2Y", "DTWEXBGS", "BAA10Y"
]

# Fair-comparison aligned task setup based on teammate's RF best settings
TASKS = {
    "Regression": {
        "type": "reg",
        "filters": ["EMA10"],
        "lookbacks": [42],
        "horizons": [30],
        "thresholds": [None]
    },
    "Binary": {
        "type": "bin",
        "filters": ["SP500", "SMA5", "EMA20"],
        "lookbacks": [54],
        "horizons": [25],
        "thresholds": [None]
    },
    "Multiclass": {
        "type": "multi",
        "filters": ["SP500", "SMA5", "SMA10", "SMA20", "EMA5", "EMA10", "EMA20"],
        "lookbacks": [48],
        "horizons": [5],
        "thresholds": [0.05]
    }
}

# Decision Tree hyperparameter search space
DEPTH_RANGE = [4, 6, 8, 10, 12, 15]
MIN_LEAF_RANGE = [1, 2, 5, 10]
MIN_SPLIT_RANGE = [2, 5, 10, 20]

TEST_RATIO = 0.2

# ============================================================
# 2) Data preparation
# ============================================================
def prepare_data(df, filter_col, macro_features, lookback, horizon, task_type, threshold=None):
    """
    Build sliding-window features aligned with teammate's RF task setup.
    """
    # Map the user-facing "SP500" option to the real column name
    feature_col = "SP500 Close" if filter_col == "SP500" else filter_col

    # Avoid duplicate columns
    selected_cols = [feature_col] + macro_features

    # Build a unique list of required columns
    required_cols = ["Date", "SP500 Close"] + [c for c in selected_cols if c != "SP500 Close"]

    temp_df = df[required_cols].copy()

    # Convert numeric columns safely
    numeric_cols = [c for c in required_cols if c != "Date"]
    for c in numeric_cols:
        temp_df[c] = pd.to_numeric(temp_df[c], errors="coerce")

    temp_df = temp_df.dropna().reset_index(drop=True)

    # Forward return always based on SP500 Close
    temp_df["FWD_RETURN"] = temp_df["SP500 Close"].shift(-horizon) / temp_df["SP500 Close"] - 1
    temp_df = temp_df.dropna().reset_index(drop=True)

    # Actual feature columns used for modeling
    model_feature_cols = [feature_col] + macro_features

    X_list, y_list = [], []
    num_rows = len(temp_df)

    for start in range(0, num_rows - lookback + 1):
        end = start + lookback

        X_chunk = temp_df.iloc[start:end][model_feature_cols].to_numpy().flatten()
        fwd_return = temp_df.loc[end - 1, "FWD_RETURN"]

        X_list.append(X_chunk)

        if task_type == "reg":
            y_list.append(fwd_return)
        elif task_type == "bin":
            y_list.append(1 if fwd_return > 0 else 0)
        elif task_type == "multi":
            if fwd_return > threshold:
                y_val = 1
            elif fwd_return < -threshold:
                y_val = -1
            else:
                y_val = 0
            y_list.append(y_val)

    return np.array(X_list), np.array(y_list)


def split_data_randomly(X, y):
    """
    Random split:
    60% train, 20% val, 20% test
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,   # 0.25 of 0.8 = 0.2 total
        random_state=RANDOM_STATE,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X, y, task_type):
    pred = model.predict(X)
    if task_type == "reg":
        return r2_score(y, pred)
    return accuracy_score(y, pred)

# ============================================================
# 3) Main experiment loop
# ============================================================
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
df_raw = df_raw.sort_values("Date").reset_index(drop=True)

all_best_models = []

for task_name, config in TASKS.items():
    print(f"\n🚀 Starting task: {task_name}")

    metric_name = "R2" if config["type"] == "reg" else "Acc"
    best_task_res = None

    total_combos = (
        len(config["filters"])
        * len(config["lookbacks"])
        * len(config["horizons"])
        * len(config["thresholds"])
        * len(DEPTH_RANGE)
        * len(MIN_LEAF_RANGE)
        * len(MIN_SPLIT_RANGE)
    )

    pbar = tqdm(total=total_combos, desc=f"Searching {task_name}")

    for f_col in config["filters"]:
        for lb in config["lookbacks"]:
            for hz in config["horizons"]:
                for th in config["thresholds"]:

                    X, y = prepare_data(
                        df=df_raw,
                        filter_col=f_col,
                        macro_features=MACRO_FEATURES,
                        lookback=lb,
                        horizon=hz,
                        task_type=config["type"],
                        threshold=th
                    )

                    if len(X) < 100:
                        pbar.update(len(DEPTH_RANGE) * len(MIN_LEAF_RANGE) * len(MIN_SPLIT_RANGE))
                        continue

                    X_train, X_val, X_test, y_train, y_val, y_test = split_data_randomly(X, y)

                    for d in DEPTH_RANGE:
                        for ml in MIN_LEAF_RANGE:
                            for ms in MIN_SPLIT_RANGE:
                                pbar.update(1)

                                if config["type"] == "reg":
                                    tree = DecisionTreeRegressor(
                                        max_depth=d,
                                        min_samples_leaf=ml,
                                        min_samples_split=ms,
                                        random_state=RANDOM_STATE
                                    )
                                else:
                                    tree = DecisionTreeClassifier(
                                        max_depth=d,
                                        min_samples_leaf=ml,
                                        min_samples_split=ms,
                                        class_weight="balanced",
                                        random_state=RANDOM_STATE
                                    )

                                model = Pipeline([
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("tree", tree)
                                ])

                                model.fit(X_train, y_train)

                                train_s = evaluate_model(model, X_train, y_train, config["type"])
                                val_s = evaluate_model(model, X_val, y_val, config["type"])
                                gap = train_s - val_s

                                if best_task_res is None or (
                                    val_s > best_task_res["Val_Score"] and gap < 0.20
                                ):
                                    best_task_res = {
                                        "Task": task_name,
                                        "Filter": f_col,
                                        "Lookback": lb,
                                        "Horizon": hz,
                                        "Threshold": th,
                                        "Depth": d,
                                        "MinLeaf": ml,
                                        "MinSplit": ms,
                                        "Train_Score": train_s,
                                        "Val_Score": val_s,
                                        "Gap": gap,
                                        "X_test": X_test,
                                        "y_test": y_test,
                                        "model": model
                                    }

    pbar.close()

    if best_task_res is not None:
        test_s = evaluate_model(
            best_task_res["model"],
            best_task_res["X_test"],
            best_task_res["y_test"],
            config["type"]
        )
        best_task_res["Test_Score"] = test_s
        all_best_models.append(best_task_res)

        print(
            f"✅ {task_name} finished! "
            f"Best Filter: {best_task_res['Filter']} | "
            f"Lookback: {best_task_res['Lookback']} | "
            f"Horizon: {best_task_res['Horizon']} | "
            f"Threshold: {best_task_res['Threshold']} | "
            f"Depth: {best_task_res['Depth']} | "
            f"MinLeaf: {best_task_res['MinLeaf']} | "
            f"MinSplit: {best_task_res['MinSplit']} | "
            f"Test {metric_name}: {test_s:.4f}"
        )
    else:
        print(f"⚠️ No valid model found for {task_name}")

# ============================================================
# 4) Final summary
# ============================================================
print("\n" + "=" * 70)
print("Aligned Decision Tree summary (fair comparison against teammate's RF setup)")
final_df = pd.DataFrame(all_best_models).drop(columns=["X_test", "y_test", "model"])
print(final_df.to_string(index=False))
