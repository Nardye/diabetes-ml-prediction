"""
diabetes_model.py
-----------------
Reusable functions for the Diabetes Hospital Readmission project.

The notebook calls these functions for the analysis, while the EDA,
charts, and written explanations stay in the notebook.

Typical workflow
----------------
    import diabetes_model as dm

    df = dm.load_data("diabetes_clean.csv")
    df = dm.clean_data(df)
    X_train, X_test, y_train, y_test = dm.prepare_data(df)

    models = {
        name: dm.train_model(name, X_train, y_train)
        for name in ["logistic", "logistic_balanced", "random_forest"]
    }

    cv_table = dm.cross_validate_models(models, X_train, y_train)   # select with CV
    grid = dm.tune_model("random_forest", X_train, y_train)         # tune with CV
    final_table = dm.evaluate_models(                               # test set, once
        {**models, "random_forest_tuned": grid.best_estimator_},
        X_test, y_test,
    )
"""

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "readmission_flag"

# Columns removed during cleaning, with the reason for each.
HIGH_MISSING_COLUMNS = ["weight", "max_glu_serum", "A1Cresult"]  # mostly missing
IDENTIFIER_COLUMNS = ["rowID"]                                    # no predictive meaning
DUPLICATE_TARGET_COLUMNS = ["readmitted"]                         # same info as the target
LEAKAGE_PRONE_COLUMNS = [                                         # computed on the full dataset
    "age_avg_los",
    "los_difference",
    "los_vs_age_average",
]
DIAGNOSIS_DESCRIPTION_COLUMNS = ["diag_1_desc", "diag_2_desc", "diag_3_desc"]  # codes are kept


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path):
    """Load the dataset from a CSV file and report its shape."""
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_data(
    df,
    high_missing_columns=HIGH_MISSING_COLUMNS,
    exclude_values=None,
):
    """
    Remove columns that should not be used for modeling.

    This function only makes decisions that do not depend on the
    train/test split (dropping identifiers, duplicate targets, and columns
    with extreme missingness). Anything learned from the data, such as
    imputation values or constant columns, happens after the split.

    Parameters
    ----------
    df : DataFrame
        Raw dataset.
    high_missing_columns : list of str
        Columns with so many missing values that they are dropped.
    exclude_values : dict, optional
        Encounters to remove, as {column_name: [values_to_remove]}.
        Example: {"discharge_disposition_id": ["Expired"]} removes patients
        who died in hospital, since they cannot be readmitted.
    """
    df_clean = df.copy()  # keep the original DataFrame unchanged

    if exclude_values:
        for column, values in exclude_values.items():
            if column not in df_clean.columns:
                warnings.warn(f"exclude_values: column '{column}' not found, skipped.")
                continue
            before = len(df_clean)
            df_clean = df_clean[~df_clean[column].isin(values)]
            print(f"Removed {before - len(df_clean)} rows where {column} in {values}")

    columns_to_drop = (
        list(high_missing_columns)
        + IDENTIFIER_COLUMNS
        + DUPLICATE_TARGET_COLUMNS
        + LEAKAGE_PRONE_COLUMNS
        + DIAGNOSIS_DESCRIPTION_COLUMNS
    )
    # Only drop columns that exist, so the function does not crash on re-runs
    columns_to_drop = [c for c in columns_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=columns_to_drop)

    print("Dataset shape after cleaning:", df_clean.shape)
    return df_clean


# ---------------------------------------------------------------------------
# 3. Prepare
# ---------------------------------------------------------------------------
def prepare_data(df, target=TARGET, test_size=0.20, random_state=RANDOM_STATE):
    """
    Split the data into features/target and into train/test sets.

    The split happens BEFORE any step that learns from the data. Constant
    columns are identified from the training set only, so no information
    from the test set influences preprocessing.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep the same readmission rate in both sets
    )

    # Constant columns carry no information; detect them on training data only
    constant_columns = [
        c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        print("Dropping constant columns:", constant_columns)
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """
    Build the preprocessing step (impute, scale, one-hot encode).

    Column types are read from X_train. When this preprocessor is placed
    inside a Pipeline, it is fit on training data only (and on the training
    folds only during cross-validation), which prevents data leakage.
    """
    numerical_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
def get_classifier(name, **params):
    """Return an unfitted classifier by name. Extra params override defaults."""
    classifiers = {
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "logistic_balanced": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(classifiers)}")
    return classifiers[name]().set_params(**params)


def make_pipeline(name, X_train, **params):
    """Preprocessing + classifier in one unfitted Pipeline."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", get_classifier(name, **params)),
    ])


def train_model(name, X_train, y_train, **params):
    """
    Build and fit a pipeline for the chosen model.

    name : "logistic", "logistic_balanced", or "random_forest"
    params : optional classifier settings, e.g. max_depth=10
    """
    pipeline = make_pipeline(name, X_train, **params)
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(model, X, y, threshold=0.5):
    """
    Calculate the five classification metrics for one fitted model.

    Returns a dict. Predictions use `threshold` on the predicted
    probability of readmission (0.5 matches model.predict()).
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_prob),
    }


def evaluate_models(models, X, y, threshold=0.5):
    """
    Evaluate several fitted models and return ONE comparison table.

    models : dict of {display_name: fitted_model}
    """
    rows = {name: evaluate_model(m, X, y, threshold) for name, m in models.items()}
    return pd.DataFrame(rows).T.round(3)


def cross_validate_models(models, X_train, y_train, n_splits=5):
    """
    Compare models with stratified cross-validation on the TRAINING data.

    Use this table to choose a model; keep the test set for the final check.
    Returns mean and standard deviation of each metric across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1 Score": "f1",
        "ROC-AUC": "roc_auc",
    }
    rows = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        row = {}
        for label in scoring:
            values = scores[f"test_{label}"]
            row[f"{label} (mean)"] = values.mean()
            row[f"{label} (std)"] = values.std()
        rows[name] = row
    return pd.DataFrame(rows).T.round(3)


# ---------------------------------------------------------------------------
# 6. Tune
# ---------------------------------------------------------------------------
DEFAULT_RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200],          # more trees = more stable, slower
    "classifier__max_depth": [None, 10, 20],         # limits overfitting
    "classifier__min_samples_split": [2, 5],         # larger = simpler trees
    "classifier__class_weight": [None, "balanced"],  # handles class imbalance
}


def tune_model(
    name,
    X_train,
    y_train,
    param_grid=DEFAULT_RF_PARAM_GRID,
    scoring="roc_auc",
    n_splits=5,
):
    """
    Tune hyperparameters with GridSearchCV using the training data only.

    Returns the fitted GridSearchCV object. Useful attributes:
        .best_params_, .best_score_, .best_estimator_, .cv_results_
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=make_pipeline(name, X_train),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------
def get_feature_importance(model, top_n=15):
    """
    Return the top_n Random Forest feature importances as a DataFrame.

    Importance shows which features the model relied on for prediction.
    It does not show that a feature causes readmission.
    """
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    result = pd.DataFrame({"Feature": names, "Importance": importances})
    result["Feature"] = (
        result["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    return result.sort_values("Importance", ascending=False).head(top_n)