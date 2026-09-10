
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🏝️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/tourism_master_dataset.csv")

@st.cache_resource
def load_models():
    return (
        joblib.load("models/rating_regressor.joblib"),
        joblib.load("models/visit_mode_classifier.joblib")
    )

df = load_data()
rating_model, mode_model = load_models()

st.title("🏝️ Tourism Experience Analytics")
st.caption("Classification, Rating Prediction & Personalized Attraction Recommendations")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a module",
        ["Dashboard", "Rating Prediction", "Visit Mode Prediction", "Recommendations"]
    )

if page == "Dashboard":
    st.header("Tourism Analytics Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Users", f"{df['UserId'].nunique():,}")
    c3.metric("Attractions", f"{df['AttractionId'].nunique():,}")
    c4.metric("Average Rating", f"{df['Rating'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visit Modes")
                with col1:
            st.subheader("Visit Modes")

            if "VisitMode_y" in df.columns:
                st.bar_chart(df["VisitMode_y"].value_counts())
            elif "VisitModeName" in df.columns:
                st.bar_chart(df["VisitModeName"].value_counts())
            elif "VisitMode" in df.columns:
                st.bar_chart(df["VisitMode"].value_counts())
            else:
                st.info("Visit mode data is not available.")
    with col2:
        st.subheader("Top Attraction Types")
        st.bar_chart(df["AttractionType"].value_counts().head(10))

    st.subheader("Top Attractions by Number of Ratings")
    top = df.groupby("Attraction").agg(
        Ratings=("Rating","count"),
        Average_Rating=("Rating","mean")
    ).sort_values("Ratings", ascending=False).head(10)
    st.dataframe(top, use_container_width=True)

elif page == "Rating Prediction":
    st.header("⭐ Attraction Rating Prediction")
    row = {
        "VisitYear": st.number_input("Visit Year", int(df.VisitYear.min()), int(df.VisitYear.max()), int(df.VisitYear.median())),
        "VisitMonth": st.number_input("Visit Month", 1, 12, 6),
        "UserContinent": st.selectbox("User Continent", sorted(df.UserContinent.dropna().astype(str).unique())),
        "UserRegion": st.selectbox("User Region", sorted(df.UserRegion.dropna().astype(str).unique())),
        "UserCountry": st.selectbox("User Country", sorted(df.UserCountry.dropna().astype(str).unique())),
        "AttractionType": st.selectbox("Attraction Type", sorted(df.AttractionType.dropna().astype(str).unique())),
        "AttractionCityName": st.selectbox("Attraction City", sorted(df.AttractionCityName.dropna().astype(str).unique()))
    }
    if st.button("Predict Rating", type="primary"):
        pred = float(rating_model.predict(pd.DataFrame([row]))[0])
        st.success(f"Predicted rating: {np.clip(pred,1,5):.2f} / 5")

elif page == "Visit Mode Prediction":
    st.header("👥 Visit Mode Prediction")
    row = {
        "VisitYear": st.number_input("Visit Year", int(df.VisitYear.min()), int(df.VisitYear.max()), int(df.VisitYear.median())),
        "VisitMonth": st.number_input("Visit Month", 1, 12, 6),
        "UserContinent": st.selectbox("Continent", sorted(df.UserContinent.dropna().astype(str).unique())),
        "UserRegion": st.selectbox("Region", sorted(df.UserRegion.dropna().astype(str).unique())),
        "UserCountry": st.selectbox("Country", sorted(df.UserCountry.dropna().astype(str).unique())),
        "AttractionType": st.selectbox("Attraction Type", sorted(df.AttractionType.dropna().astype(str).unique())),
        "AttractionCityName": st.selectbox("Attraction City", sorted(df.AttractionCityName.dropna().astype(str).unique()))
    }
    if st.button("Predict Visit Mode", type="primary"):
        pred = mode_model.predict(pd.DataFrame([row]))[0]
        st.success(f"Predicted visit mode: {pred}")

elif page == "Recommendations":
    st.header("📍 Personalized Attraction Recommendations")
    attr = df.groupby(
        ["AttractionId","Attraction","AttractionCityName","AttractionType"],
        dropna=False
    ).agg(
        AvgRating=("Rating","mean"),
        RatingCount=("Rating","count")
    ).reset_index()

    selected_type = st.selectbox(
        "Preferred Attraction Type",
        ["Any"] + sorted(attr["AttractionType"].dropna().astype(str).unique())
    )
    selected_city = st.selectbox(
        "Preferred Attraction City",
        ["Any"] + sorted(attr["AttractionCityName"].dropna().astype(str).unique())
    )
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    candidates = attr.copy()
    if selected_type != "Any":
        candidates = candidates[candidates["AttractionType"].astype(str).eq(selected_type)]
    if selected_city != "Any":
        candidates = candidates[candidates["AttractionCityName"].astype(str).eq(selected_city)]

    if candidates.empty:
        st.warning("No attractions match the selected filters.")
    else:
        result = candidates.sort_values(
            ["AvgRating","RatingCount"], ascending=[False,False]
        ).head(top_n)
        result = result.rename(columns={
            "Attraction":"Attraction Name",
            "AttractionCityName":"City",
            "AttractionType":"Type",
            "AvgRating":"Average Rating",
            "RatingCount":"Ratings"
        })
        st.dataframe(result[["Attraction Name","City","Type","Average Rating","Ratings"]],
                     use_container_width=True)

st.divider()
st.caption("Developed as a Tourism Experience Analytics portfolio project.")
