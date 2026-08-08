"""
Streamlit app: Personalized Car Recommendation System
=======================================================
Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder to a GitHub repo (app.py, recommender.py,
       requirements.txt, cardekho_dataset/cardekho_dataset.csv).
    2. Go to https://share.streamlit.io -> "New app" -> point it at the repo
       and set the main file to app.py.
    3. Done — no secrets or extra config needed for the default dataset.

Users can also upload their own CSV
via the sidebar; the app falls back to the downloaded kaggle dataset otherwise.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from recommender import (
    DEFAULT_COLUMN_MAP,
    UserPreferences,
    PersonalizedCarRecommender,
    clean_data,
    build_feature_pipeline,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Recommender",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIORITY_LABELS = {
    "mileage": "Fuel efficiency (mileage)",
    "power": "Performance (power)",
    "price": "Lower price",
    "low_km": "Lower odometer (km driven)",
    "newer": "Newer vehicle (lower age)",
}


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    import os
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "cardekho_dataset", "cardekho_dataset.csv"))


@st.cache_data(show_spinner=False)
def get_clean_data(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    return clean_data(df, col_map)


@st.cache_resource(show_spinner=False)
def get_recommender(_clean_df: pd.DataFrame, col_map: dict):
    """Builds the feature pipeline + recommender once, cached across reruns.
    Prefixing the df argument with `_` tells Streamlit not to hash it (it's
    already keyed indirectly via col_map + the data-loading cache above)."""
    feature_matrix, preprocessor, num_feats, cat_feats = build_feature_pipeline(_clean_df, col_map)
    model = PersonalizedCarRecommender(
        dataframe=_clean_df,
        feature_matrix=feature_matrix,
        preprocessor=preprocessor,
        col_map=col_map,
        numeric_features=num_feats,
        categorical_features=cat_feats,
    )
    return model


# ---------------------------------------------------------------------------
# Sidebar — data source + column mapping check
# ---------------------------------------------------------------------------

st.sidebar.title("🚗 Car Recommender")
st.sidebar.caption("Content-based recommendation engine")

with st.sidebar.expander("📁 Data source", expanded=False):
    uploaded_file = st.file_uploader("Upload your own CSV (optional)", type=["csv"])
    st.caption("No upload needed — a sample dataset is bundled by default.")

try:
    raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Couldn't read the uploaded file: {e}")
    st.stop()

col_map = DEFAULT_COLUMN_MAP
missing_cols = [v for v in col_map.values() if v not in raw_df.columns]
if missing_cols:
    st.sidebar.warning(
        f"⚠️ Expected columns not found: {missing_cols}. "
        "Using the bundled sample dataset schema instead."
    )
    st.error(
        "The uploaded file's columns don't match the expected schema "
        f"({missing_cols} missing). Expected columns: {list(col_map.values())}."
    )
    st.stop()

with st.spinner("Cleaning data and building the recommendation model..."):
    clean_df = get_clean_data(raw_df, col_map)
    if len(clean_df) < 5:
        st.error("Not enough valid rows remain after cleaning. Check your CSV's data quality.")
        st.stop()
    model = get_recommender(clean_df, col_map)

st.sidebar.success(f"✅ {len(clean_df):,} listings loaded")

# ---------------------------------------------------------------------------
# Sidebar — user preference inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Your Preferences")

price_min_data = int(clean_df[col_map["price"]].min())
price_max_data = int(clean_df[col_map["price"]].max())
budget_min, budget_max = st.sidebar.slider(
    "Budget range",
    min_value=price_min_data,
    max_value=price_max_data,
    value=(price_min_data, price_max_data),
    step=max(1, (price_max_data - price_min_data) // 100),
)

fuel_options = sorted(clean_df[col_map["fuel_type"]].dropna().unique().tolist())
preferred_fuel_types = st.sidebar.multiselect("Fuel type", fuel_options, default=[])

trans_options = sorted(clean_df[col_map["transmission"]].dropna().unique().tolist())
preferred_transmission = st.sidebar.multiselect("Transmission", trans_options, default=[])

min_seats = st.sidebar.slider("Minimum seats", 2, 10, 5)

max_km_data = int(clean_df[col_map["km_driven"]].max())
max_km_driven = st.sidebar.slider("Max km driven", 0, max_km_data, max_km_data)

max_age_data = int(clean_df[col_map["vehicle_age"]].max())
max_vehicle_age = st.sidebar.slider("Max vehicle age (years)", 0, max_age_data, max_age_data)

brand_options = sorted(clean_df[col_map["brand"]].dropna().unique().tolist())
preferred_brands = st.sidebar.multiselect("Preferred brand(s)", brand_options, default=[])

st.sidebar.markdown("**Rank what matters most** (top = most important)")
priority_choices = st.sidebar.multiselect(
    "Priorities, in order",
    options=list(PRIORITY_LABELS.keys()),
    default=["price", "mileage", "low_km"],
    format_func=lambda k: PRIORITY_LABELS[k],
)

top_n = st.sidebar.slider("Number of recommendations", 3, 20, 5)

similarity_weight = st.sidebar.slider(
    "Similarity vs. priority weighting",
    0.0, 1.0, 0.5,
    help="0 = rank purely by your stated priorities. 1 = rank purely by similarity to an 'ideal car' built from your inputs.",
)

prefs = UserPreferences(
    budget_min=budget_min,
    budget_max=budget_max,
    preferred_fuel_types=preferred_fuel_types,
    preferred_transmission=preferred_transmission,
    min_seats=min_seats,
    max_km_driven=max_km_driven,
    max_vehicle_age=max_vehicle_age,
    priorities=priority_choices,
    preferred_brands=preferred_brands,
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Personalized Car Recommendation System")
st.caption("Content-based filtering (cosine similarity) + rule-based preference matching")

tab_recs, tab_explore, tab_about = st.tabs(["🎯 Recommendations", "📊 Explore the Data", "ℹ️ About"])

# ---- Recommendations tab ---------------------------------------------------
with tab_recs:
    recs = model.recommend(prefs, top_n=top_n, similarity_weight=similarity_weight)

    if len(recs) == 0:
        st.warning("No cars match these filters. Try widening your budget or relaxing constraints.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Matches found", len(recs))
        m2.metric("Avg. match score", f"{recs['match_score'].mean():.0%}")
        m3.metric(
            "Price range",
            f"₹{recs[col_map['price']].min():,.0f} – ₹{recs[col_map['price']].max():,.0f}"
        )

        st.markdown("### Top Matches")
        for i, (_, row) in enumerate(recs.iterrows(), start=1):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{i}. {row[col_map['name']]}** — {row[col_map['brand']]}")
                    st.caption(model.explain(row, prefs))
                    badge_cols = st.columns(4)
                    badge_cols[0].markdown(f"⛽ {row[col_map['fuel_type']]}")
                    badge_cols[1].markdown(f"⚙️ {row[col_map['transmission']]}")
                    badge_cols[2].markdown(f"🛣️ {row[col_map['km_driven']]:,.0f} km")
                    badge_cols[3].markdown(f"📅 {row[col_map['vehicle_age']]:.0f} yrs old")
                with c2:
                    st.metric("Price", f"₹{row[col_map['price']]:,.0f}")
                    st.progress(min(1.0, float(row["match_score"])), text=f"{row['match_score']:.0%} match")

        st.markdown("### Where these sit in the market")
        fig = px.scatter(
            clean_df, x=col_map["km_driven"], y=col_map["price"],
            opacity=0.25, color_discrete_sequence=["gray"],
            labels={col_map["km_driven"]: "KM Driven", col_map["price"]: "Price"},
        )
        fig.add_scatter(
            x=recs[col_map["km_driven"]], y=recs[col_map["price"]],
            mode="markers", marker=dict(size=12, color="crimson"),
            name="Recommended",
        )
        fig.update_layout(showlegend=True, height=420)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show as a table"):
            display_cols = [col_map[k] for k in
                             ("name", "brand", "price", "fuel_type", "transmission", "km_driven", "vehicle_age")
                             if col_map[k] in recs.columns] + ["match_score"]
            st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)

        csv_bytes = recs[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download recommendations as CSV", csv_bytes, "recommendations.csv", "text/csv")

# ---- Explore tab -----------------------------------------------------------
with tab_explore:
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total listings", f"{len(clean_df):,}")
    c2.metric("Brands", clean_df[col_map["brand"]].nunique())
    c3.metric("Avg. price", f"₹{clean_df[col_map['price']].mean():,.0f}")
    c4.metric("Avg. km driven", f"{clean_df[col_map['km_driven']].mean():,.0f}")

    fig1 = px.histogram(clean_df, x=col_map["price"], nbins=40, title="Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fuel_counts = clean_df[col_map["fuel_type"]].value_counts().reset_index()
        fuel_counts.columns = ["fuel_type", "count"]
        fig2 = px.bar(fuel_counts, x="fuel_type", y="count", title="Listings by Fuel Type")
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        brand_counts = clean_df[col_map["brand"]].value_counts().head(10).reset_index()
        brand_counts.columns = ["brand", "count"]
        fig3 = px.bar(brand_counts, x="count", y="brand", orientation="h", title="Top 10 Brands")
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(
        clean_df, x=col_map["vehicle_age"], y=col_map["price"],
        opacity=0.3, title="Price vs. Vehicle Age",
        labels={col_map["vehicle_age"]: "Vehicle Age (yrs)", col_map["price"]: "Price"},
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---- About tab --------------------------------------------------------------
with tab_about:
    st.markdown("""
### How this works
This is a **hybrid content-based recommender**:

1. **Hard filters** — budget, fuel type, transmission, minimum seats, max km/age are applied first;
   a listing is never shown if it violates a stated constraint.
2. **Feature engineering** — numeric columns (price, mileage, power, km driven, age, seats) are
   standardized; categorical columns (fuel type, transmission, brand) are one-hot encoded.
3. **"Ideal car" vector** — a synthetic feature vector is built from your stated preferences and
   priorities.
4. **Ranking** — surviving listings are scored by a blend of cosine similarity to that ideal vector
   and a weighted score across your ranked priorities, controlled by the *similarity vs. priority
   weighting* slider in the sidebar.

### Notes on evaluation
There's no user-rating ground truth in most used-car datasets, so accuracy/precision numbers aren't
meaningful here — this is a similarity/rule-based system, not a trained classifier. See the
companion analysis notebook for coverage, diversity, and filter-correctness checks used instead.

### Deployment
This app is built to deploy as-is on **Streamlit Community Cloud**: it bundles a sample dataset so
it works with zero configuration, and accepts a user-uploaded CSV as an alternative data source.
""")
