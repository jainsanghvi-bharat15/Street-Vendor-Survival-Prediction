from pathlib import Path
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="Street Vendor Survival Analysis",
    page_icon="📊",
    layout="wide",
)


# PROJECT PATHS
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "processed_dataset"
MODEL_DIR = ROOT / "models"

# Supporting the original trained model and the optional renamed copy.
MODEL_CANDIDATES = [
    ROOT / "trained_model.pkl",
    MODEL_DIR / "trained_model.pkl",
    MODEL_DIR / "vendor_survival_model.joblib",
]

FEATURES_PATH = MODEL_DIR / "model_features.json"
PREFERRED_CSV_NAMES = [
    "street_vendor_survival_cleaned.csv",
    "street_vendor_survival_cleaned_dataset.csv",
    "cleaned_dataset.csv",
]

# DATASET LOADING
def locate_dataset():
    """Find the most likely cleaned dataset for the overview pages."""

    if not DATA_DIR.exists():
        return None

    for filename in PREFERRED_CSV_NAMES:
        candidate = DATA_DIR / filename

        if candidate.is_file():
            return candidate

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if len(csv_files) == 1:
        return csv_files[0]

    cleaned_files = [
        path
        for path in csv_files
        if "clean" in path.stem.lower()
        and "feature" not in path.stem.lower()
        and "engineer" not in path.stem.lower()
    ]

    if len(cleaned_files) == 1:
        return cleaned_files[0]

    return None


@st.cache_data(show_spinner="Loading vendor dataset...")
def load_data(path_string, modified_time):
    """Load a CSV and refresh the cache when the file changes."""

    return pd.read_csv(path_string)


@st.cache_data(show_spinner="Checking datasets for model features...")
def load_csv_for_prediction(path_string, modified_time):
    """Load a candidate CSV for checking model compatibility."""

    return pd.read_csv(path_string)


# MODEL LOADING
def locate_model():
    """Return the first supported model file that exists."""

    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None

@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path_string, modified_time):
    """
     Load a trusted local model.
    .pkl files are loaded using pickle.
    .joblib files are loaded using joblib.
    """

    model_path = Path(model_path_string)
    if model_path.suffix.lower() == ".pkl":
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    else:
        model = joblib.load(model_path)

    return model


def get_model_features(model):
    """
    Obtaining the model's feature names.
    Prefering feature names saved by the trained model.
    Falling back to model_features.json if available.
    """

    if hasattr(model, "feature_names_in_"):
        features = [
            str(feature)
            for feature in model.feature_names_in_
        ]

        if features:
            return features

    if FEATURES_PATH.is_file():
        with open(FEATURES_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        features = metadata.get("features", [])

        if features:
            return [str(feature) for feature in features]

    return None

# FINDING A CSV CONTAINING ALL MODEL FEATURES
def find_compatible_dataset(features):
    """
    Search CSV files for one containing every required model feature.
    This allows the prediction page to use the feature-engineered
    dataset while the overview pages continue using the cleaned CSV.
    """

    if not DATA_DIR.exists():
        return None, None, []

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        return None, None, []

    compatible_files = []
    problems = []

    for csv_path in csv_files:
        try:
            candidate_df = load_csv_for_prediction(
                str(csv_path),
                csv_path.stat().st_mtime,
            )

            missing_features = [
                feature
                for feature in features
                if feature not in candidate_df.columns
            ]

            if not missing_features:
                compatible_files.append(
                    (csv_path, candidate_df)
                )

        except Exception as exc:
            problems.append(
                f"{csv_path.name}: {exc}"
            )

    if not compatible_files:
        return None, None, problems

    # Prefering a filename that indicates feature engineering.
    preferred = [
        item
        for item in compatible_files
        if any(
            word in item[0].stem.lower()
            for word in ["feature", "engineer", "encoded", "processed"]
        )
    ]

    if len(preferred) == 1:
        return preferred[0][0], preferred[0][1], problems

    # If only one dataset matches, use it.
    if len(compatible_files) == 1:
        return (
            compatible_files[0][0],
            compatible_files[0][1],
            problems,
        )

    # Multiple compatible datasets: let the user choose.
    return None, None, problems

# PAGE HEADER
st.title("Street Vendor Survival Analysis")

st.caption(
    "Student project demonstration • 12-month dataset outcome • "
    "Not a validated real-world decision system"
)


# LOADIG THE MAIN DATASET
csv_path = locate_dataset()

if csv_path is None:
    st.error("Could not identify the cleaned CSV dataset.")
    st.write("Expected folder:")
    st.code(str(DATA_DIR))
    st.write("CSV files found:")

    if DATA_DIR.exists():
        available_csvs = sorted(DATA_DIR.glob("*.csv"))

        if available_csvs:
            for file in available_csvs:
                st.write(f"- {file.name}")
        else:
            st.write("No CSV files were found.")
    else:
        st.write("The folder does not exist.")

    st.info(
        "Place your cleaned CSV inside datasets/processed_dataset. "
        "If several CSV files exist, rename the cleaned dataset to "
        "'street_vendor_survival_cleaned.csv'."
    )

    st.stop()

try:
    df = load_data(
        str(csv_path),
        csv_path.stat().st_mtime,
    )

except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.code(str(csv_path))
    st.stop()

if "vendor_survived" not in df.columns:
    st.error(
        "The selected CSV does not contain the required "
        "'vendor_survived' column."
    )

    st.write("Columns found:")
    st.write(list(df.columns))
    st.stop()

# Work on a copy. The original CSV is never modified.
df = df.copy()

target = pd.to_numeric(
    df["vendor_survived"],
    errors="coerce",
)

df["_outcome"] = (
    target.map({
        1: "Survived",
        0: "Closed",
    })
    .fillna("Unknown")
)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Dataset Overview",
            "Explore Vendors",
            "Prediction Demo",
        ],
    )

    st.divider()

    st.caption(f"Dataset: {csv_path.name}")
    st.caption(f"Records: {len(df):,}")

# PAGE 1: DATASET OVERVIEW
if page == "Dataset Overview":

    st.subheader("Dataset Overview")

    total = len(df)
    survived = int((target == 1).sum())
    closed = int((target == 0).sum())

    known_outcomes = survived + closed

    survival_rate = (
        survived / known_outcomes
        if known_outcomes
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{total:,}")
    c2.metric("Survived", f"{survived:,}")
    c3.metric("Closed", f"{closed:,}")
    c4.metric("Survival Rate", f"{survival_rate:.2%}")

    left, right = st.columns(2)

    with left:
        st.markdown("### 12-Month Outcome Distribution")

        counts = (
            df["_outcome"]
            .value_counts()
            .reindex(
                ["Survived", "Closed", "Unknown"],
                fill_value=0,
            )
        )

        st.bar_chart(counts)

    with right:
        if "city" in df.columns:
            st.markdown("### Vendor Records by City")

            city_counts = (
                df["city"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("City")
            )

            st.bar_chart(city_counts)

        else:
            st.info(
                "The 'city' column is not present in this dataset."
            )

    if "food_category" in df.columns:
        st.markdown("### Vendor Records by Food Category")

        food_counts = (
            df["food_category"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Food Category")
        )

        st.bar_chart(food_counts)

    st.info(
        "These are descriptive summaries of the supplied dataset. "
        "The target represents a recorded 12-month outcome, "
        "not a verified current vendor status."
    )


# PAGE 2: EXPLORE VENDORS
elif page == "Explore Vendors":
    st.subheader("Explore Vendor Records")
    filtered = df.copy()
    available_filters = [
        column
        for column in [
            "city",
            "zone_type",
            "food_category",
            "license_status",
        ]
        if column in filtered.columns
    ]

    if available_filters:
        filter_cols = st.columns(
            min(4, len(available_filters))
        )

        for slot, column in zip(
            filter_cols,
            available_filters,
        ):
            options = sorted(
                filtered[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            choice = slot.selectbox(
                f"Filter: {column.replace('_', ' ').title()}",
                ["All"] + options,
                key=f"filter_{column}",
            )

            if choice != "All":
                filtered = filtered[
                    filtered[column].astype(str) == choice
                ]

    outcome_choice = st.selectbox(
        "Filter by 12-month outcome",
        ["All", "Survived", "Closed", "Unknown"],
    )

    if outcome_choice != "All":
        filtered = filtered[
            filtered["_outcome"] == outcome_choice
        ]

    st.write(f"**Matching records:** {len(filtered):,}")

    preferred_columns = [
        "vendor_id",
        "city",
        "zone_type",
        "food_category",
        "license_status",
        "vendor_age_years",
        "years_in_business",
        "avg_daily_revenue_inr",
        "avg_daily_customers",
        "num_helpers",
        "hours_open_per_day",
        "competition_within_100m",
        "has_online_presence",
        "vendor_survived",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered.columns
    ]

    if not display_columns:
        display_columns = [
            column
            for column in filtered.columns
            if column != "_outcome"
        ]

    st.dataframe(
        filtered[display_columns].head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "For readability, at most 1,000 matching records are displayed."
    )

    numeric_columns = [
        column
        for column in filtered.select_dtypes(
            include=np.number
        ).columns
        if column != "vendor_survived"
    ]

    if numeric_columns:
        selected_metric = st.selectbox(
            "Optional numerical summary",
            ["None"] + numeric_columns,
        )

        if selected_metric != "None":
            st.dataframe(
                filtered[selected_metric]
                .describe()
                .to_frame("Summary"),
                use_container_width=True,
            )

# PAGE 3: PREDICTION DEMO
else:
    st.subheader("Prediction Demo")
    st.warning(
        "This is an academic demonstration. Predictions depend on "
        "the trained model and the same feature engineering used "
        "during training. Outputs are not guaranteed real-world outcomes."
    )

    model_path = locate_model()

    # Check whether a trained model exists
    if model_path is None:
        st.error("No trained model file was found.")
        st.write("The app checks these locations:")
        st.code(
            "trained_model.pkl\n"
            "models/trained_model.pkl\n"
            "models/vendor_survival_model.joblib"
        )

        st.info(
            "Place your existing trained_model.pkl file in the "
            "project root, beside app.py, or inside the models folder."
        )

    else:
        try:
            model = load_model(
                str(model_path),
                model_path.stat().st_mtime,
            )

            st.success(
                f"Trained model loaded: {model_path.name}"
            )
            # Getting the exact model feature list
            features = get_model_features(model)
            if not features:

                st.error(
                    "Could not determine the model's expected "
                    "feature names."
                )

                st.write(
                    "The model does not expose feature_names_in_, "
                    "and models/model_features.json was not available "
                    "or did not contain a feature list."
                )

                st.stop()

            st.caption(
                f"Expected model input: {len(features)} features"
            )

            # Locating a dataset with all required features
            prediction_path, prediction_df, csv_problems = (
                find_compatible_dataset(features)
            )

            if prediction_path is None:
                # Determining whether files exist but more than one matches.
                compatible_candidates = []
                if DATA_DIR.exists():
                    for candidate_path in sorted(
                        DATA_DIR.glob("*.csv")
                    ):
                        try:
                            candidate_df = load_csv_for_prediction(
                                str(candidate_path),
                                candidate_path.stat().st_mtime,
                            )

                            if all(
                                feature in candidate_df.columns
                                for feature in features
                            ):
                                compatible_candidates.append(
                                    candidate_path
                                )

                        except Exception:
                            pass

                if compatible_candidates:
                    st.info(
                        "More than one CSV contains all model features. "
                        "Select the dataset you used for model training."
                    )

                    selected_filename = st.selectbox(
                        "Choose feature-compatible dataset",
                        [
                            path.name
                            for path in compatible_candidates
                        ],
                    )

                    prediction_path = next(
                        path
                        for path in compatible_candidates
                        if path.name == selected_filename
                    )

                    prediction_df = load_csv_for_prediction(
                        str(prediction_path),
                        prediction_path.stat().st_mtime,
                    )

                else:

                    st.error(
                        "No CSV in datasets/processed_dataset contains "
                        "all the features expected by the trained model."
                    )

                    st.write(
                        "The model needs the same 63 feature columns "
                        "used during training, including any engineered "
                        "and one-hot-encoded columns."
                    )

                    st.write("Expected feature names:")

                    st.code(
                        "\n".join(features),
                        language="text",
                    )

                    st.info(
                        "Put the feature-engineered CSV used during "
                        "training in datasets/processed_dataset. "
                        "Do not use the cleaned CSV unless it contains "
                        "all the required model features."
                    )

                    if csv_problems:
                        with st.expander(
                            "Details about CSV files that could not be read"
                        ):
                            for problem in csv_problems:
                                st.write(problem)

                    st.stop()

            # Validating features
            missing_features = [
                feature
                for feature in features
                if feature not in prediction_df.columns
            ]

            if missing_features:
                st.error("The selected dataset is missing required model features.")
                st.write(missing_features)
                st.stop()

            # Keeping the model's exact feature order.
            X = prediction_df[features].copy()

            # The uploaded model is an XGBoost classifier.
            # Its feature-engineered inputs should be numeric.
            non_numeric_features = [
                column
                for column in X.columns
                if not pd.api.types.is_numeric_dtype(X[column])
            ]

            if non_numeric_features:
                st.error(
                    "Some model features are not numeric. The same "
                    "encoding used during training must be applied first.")
                st.write(non_numeric_features)
                st.stop()
            st.success(f"Compatible dataset found: {prediction_path.name}")

            st.write(
                f"**Records available for demonstration:** "
                f"{len(prediction_df):,}")

            # Selecting an existing record
            st.markdown("### Select a Vendor Record")
            record_number = st.number_input(
                "Dataset row number",
                min_value=1,
                max_value=len(prediction_df),
                value=1,
                step=1,
            )
            row_index = int(record_number) - 1
            selected_row = prediction_df.iloc[[row_index]]

            # Displaying useful vendor information where available.
            display_fields = [
                "vendor_id",
                "city",
                "zone_type",
                "food_category",
                "license_status",
                "vendor_age_years",
                "years_in_business",
                "avg_daily_revenue_inr",
                "avg_daily_customers",
                "vendor_survived",
            ]

            available_display_fields = [
                column
                for column in display_fields
                if column in selected_row.columns
            ]

            if available_display_fields:
                st.markdown("#### Selected Record Details")

                st.dataframe(
                    selected_row[available_display_fields],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "The model receives the selected record's existing "
                "feature-engineered values. This avoids guessing missing "
                "values or creating an incompatible input form."
            )

            # Running prediction
            if st.button(
                "Run Model Prediction",
                type="primary",
            ):

                try:
                    model_input = selected_row[features]
                    prediction = int(
                        np.asarray(
                            model.predict(model_input)
                        ).ravel()[0]
                    )

                    class_labels = {
                        1: "Survived",
                        0: "Closed",
                    }

                    predicted_label = class_labels.get(
                        prediction,
                        str(prediction),
                    )

                    st.markdown("### Prediction Result")
                    if prediction == 1:
                        st.success(
                            f"Predicted 12-month outcome: {predicted_label}"
                        )

                    elif prediction == 0:
                        st.error(
                            f"Predicted 12-month outcome: {predicted_label}"
                        )

                    else:
                        st.info(f"Model output: {predicted_label}")

                    # Show class probabilities when available.
                    if hasattr(model, "predict_proba"):

                        probabilities = np.asarray(
                            model.predict_proba(model_input)
                        )[0]

                        classes = getattr(
                            model,
                            "classes_",
                            range(len(probabilities)),
                        )

                        probability_rows = []

                        for class_value, probability in zip(
                            classes,
                            probabilities,
                        ):
                            class_name = class_labels.get(
                                int(class_value),
                                str(class_value),
                            )

                            probability_rows.append({
                                "Class": class_name,
                                "Probability": float(probability),
                            })

                        probability_df = pd.DataFrame(
                            probability_rows
                        )

                        st.markdown("#### Model Class Probabilities")

                        st.dataframe(
                            probability_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Comparing with the recorded outcome if available.
                    if "vendor_survived" in selected_row.columns:

                        actual_value = pd.to_numeric(
                            selected_row["vendor_survived"],
                            errors="coerce",
                        ).iloc[0]

                        if pd.notna(actual_value):

                            actual_value = int(actual_value)

                            actual_label = class_labels.get(
                                actual_value,
                                str(actual_value),
                            )

                            st.write(
                                f"**Recorded dataset outcome:** {actual_label}"
                            )

                            if actual_value == prediction:
                                st.info(
                                    "The prediction matches this record's "
                                    "stored outcome. This single-record "
                                    "comparison does not establish model accuracy."
                                )
                            else:
                                st.warning(
                                    "The prediction differs from this "
                                    "record's stored outcome."
                                )

                    st.caption(
                        "This is an experimental prediction on a selected "
                        "dataset record. It is not a guarantee of future "
                        "vendor survival. A proper evaluation on held-out "
                        "data is needed to assess generalization."
                    )

                except Exception as exc:
                    st.error("Prediction failed.")
                    st.write(
                        "Check that the model, feature-engineered dataset, "
                        "feature names, and preprocessing match the original "
                        "training pipeline."
                    )
                    st.exception(exc)

        except Exception as exc:
            st.error("Could not load or use the trained model.")
            st.write(
                "This could be due to missing dependencies, version mismatch of the "
                "packages, or the inability to load the model file."
            )

            st.exception(exc)

# ABOUT THIS PROJECT
with st.expander("About this project"):

    st.write(
        "My project involves analysis of the street vendor dataset using "
        "data preparation, exploratory data analysis, SQL, machine learning "
        "experimentation, and Power BI Desktop. This Streamlit interface "
        "offers local dataset exploration and an example of a trained model. "
        "Predictions are dependent on compatible feature engineering and "
        "training preprocessing. Results cannot be considered as valid real-life "
        "decisions.."
    )