import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🏝️",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/tourism_master_dataset.csv")


# ---------------------------------------------------------
# LOAD MACHINE LEARNING MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    rating_model = joblib.load("models/rating_regressor.joblib")
    mode_model = joblib.load("models/visit_mode_classifier.joblib")
    return rating_model, mode_model


# ---------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------
df = load_data()
rating_model, mode_model = load_models()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🏝️ Tourism Experience Analytics")
st.caption(
    "Classification, Rating Prediction & Personalized Attraction Recommendations"
)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Choose a module",
        [
            "Dashboard",
            "Rating Prediction",
            "Visit Mode Prediction",
            "Recommendations"
        ]
    )


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.header("Tourism Analytics Dashboard")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Transactions",
        f"{len(df):,}"
    )

    c2.metric(
        "Users",
        f"{df['UserId'].nunique():,}"
    )

    c3.metric(
        "Attractions",
        f"{df['AttractionId'].nunique():,}"
    )

    c4.metric(
        "Average Rating",
        f"{df['Rating'].mean():.2f}"
    )

    # Two charts
    col1, col2 = st.columns(2)

    # Visit Mode chart
    with col1:

        st.subheader("Visit Modes")

        possible_visit_mode_columns = [
            "VisitMode",
            "VisitModeLabel",
            "VisitModeName",
            "VisitMode_x",
            "VisitMode_y"
        ]

        visit_mode_col = None

        for column in possible_visit_mode_columns:
            if column in df.columns:
                visit_mode_col = column
                break

        if visit_mode_col is not None:

            visit_modes = (
                df[visit_mode_col]
                .dropna()
                .astype(str)
                .value_counts()
            )

            if not visit_modes.empty:
                st.bar_chart(visit_modes)
            else:
                st.info("No visit mode data available.")

        else:
            st.info("Visit mode data is not available in the dataset.")

    # Attraction Type chart
    with col2:

        st.subheader("Top Attraction Types")

        if "AttractionType" in df.columns:

            attraction_types = (
                df["AttractionType"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
            )

            st.bar_chart(attraction_types)

        else:
            st.info("Attraction type data is not available.")

    # Top attractions
    st.subheader("Top Attractions by Number of Ratings")

    if "Attraction" in df.columns and "Rating" in df.columns:

        top = (
            df.groupby("Attraction")
            .agg(
                Ratings=("Rating", "count"),
                Average_Rating=("Rating", "mean")
            )
            .sort_values(
                "Ratings",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top,
            use_container_width=True
        )

    else:
        st.info("Attraction rating data is not available.")


# =========================================================
# RATING PREDICTION
# =========================================================
elif page == "Rating Prediction":

    st.header("⭐ Attraction Rating Prediction")

    row = {
        "VisitYear": st.number_input(
            "Visit Year",
            int(df["VisitYear"].min()),
            int(df["VisitYear"].max()),
            int(df["VisitYear"].median())
        ),

        "VisitMonth": st.number_input(
            "Visit Month",
            1,
            12,
            6
        ),

        "UserContinent": st.selectbox(
            "User Continent",
            sorted(
                df["UserContinent"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),

        "UserRegion": st.selectbox(
            "User Region",
            sorted(
                df["UserRegion"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),

        "UserCountry": st.selectbox(
            "User Country",
            sorted(
                df["UserCountry"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),

        "AttractionType": st.selectbox(
            "Attraction Type",
            sorted(
                df["AttractionType"]
                .dropna()
                .astype(str)
                .unique()
            )
        ),

        "AttractionCityName": st.selectbox(
            "Attraction City",
            sorted(
                df["AttractionCityName"]
                .dropna()
                .astype(str)
                .unique()
            )
        )
    }

    if st.button(
        "Predict Rating",
        type="primary"
    ):

        try:

            input_data = pd.DataFrame([row])

            prediction = rating_model.predict(input_data)[0]

            prediction = float(
                np.clip(
                    prediction,
                    1,
                    5
                )
            )

            st.success(
                f"Predicted rating: {prediction:.2f} / 5"
            )

        except Exception as e:

            st.error(
                "Unable to generate the rating prediction."
            )

            st.exception(e)


# =========================================================
# VISIT MODE PREDICTION
# =========================================================
elif page == "Visit Mode Prediction":

    st.header("👥 Visit Mode Prediction")

    row = {
        "VisitYear": st.number_input(
            "Visit Year",
            int(df["VisitYear"].min()),
            int(df["VisitYear"].max()),
            int(df["VisitYear"].median()),
            key="mode_year"
        ),

        "VisitMonth": st.number_input(
            "Visit Month",
            1,
            12,
            6,
            key="mode_month"
        ),

        "UserContinent": st.selectbox(
            "Continent",
            sorted(
                df["UserContinent"]
                .dropna()
                .astype(str)
                .unique()
            ),
            key="mode_continent"
        ),

        "UserRegion": st.selectbox(
            "Region",
            sorted(
                df["UserRegion"]
                .dropna()
                .astype(str)
                .unique()
            ),
            key="mode_region"
        ),

        "UserCountry": st.selectbox(
            "Country",
            sorted(
                df["UserCountry"]
                .dropna()
                .astype(str)
                .unique()
            ),
            key="mode_country"
        ),

        "AttractionType": st.selectbox(
            "Attraction Type",
            sorted(
                df["AttractionType"]
                .dropna()
                .astype(str)
                .unique()
            ),
            key="mode_type"
        ),

        "AttractionCityName": st.selectbox(
            "Attraction City",
            sorted(
                df["AttractionCityName"]
                .dropna()
                .astype(str)
                .unique()
            ),
            key="mode_city"
        )
    }

    if st.button(
        "Predict Visit Mode",
        type="primary"
    ):

        try:

            input_data = pd.DataFrame([row])

            prediction = mode_model.predict(input_data)[0]

            st.success(
                f"Predicted visit mode: {prediction}"
            )

        except Exception as e:

            st.error(
                "Unable to generate the visit mode prediction."
            )

            st.exception(e)


# =========================================================
# RECOMMENDATIONS
# =========================================================
elif page == "Recommendations":

    st.header("📍 Personalized Attraction Recommendations")

    attr = (
        df.groupby(
            [
                "AttractionId",
                "Attraction",
                "AttractionCityName",
                "AttractionType"
            ],
            dropna=False
        )
        .agg(
            AvgRating=("Rating", "mean"),
            RatingCount=("Rating", "count")
        )
        .reset_index()
    )

    selected_type = st.selectbox(
        "Preferred Attraction Type",
        [
            "Any"
        ]
        + sorted(
            attr["AttractionType"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    selected_city = st.selectbox(
        "Preferred Attraction City",
        [
            "Any"
        ]
        + sorted(
            attr["AttractionCityName"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    top_n = st.slider(
        "Number of recommendations",
        3,
        10,
        5
    )

    candidates = attr.copy()

    if selected_type != "Any":

        candidates = candidates[
            candidates["AttractionType"]
            .astype(str)
            .eq(selected_type)
        ]

    if selected_city != "Any":

        candidates = candidates[
            candidates["AttractionCityName"]
            .astype(str)
            .eq(selected_city)
        ]

    if candidates.empty:

        st.warning(
            "No attractions match the selected filters."
        )

    else:

        result = (
            candidates
            .sort_values(
                [
                    "AvgRating",
                    "RatingCount"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .head(top_n)
        )

        result = result.rename(
            columns={
                "Attraction": "Attraction Name",
                "AttractionCityName": "City",
                "AttractionType": "Type",
                "AvgRating": "Average Rating",
                "RatingCount": "Ratings"
            }
        )

        st.dataframe(
            result[
                [
                    "Attraction Name",
                    "City",
                    "Type",
                    "Average Rating",
                    "Ratings"
                ]
            ],
            use_container_width=True
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Developed as a Tourism Experience Analytics portfolio project."
)
