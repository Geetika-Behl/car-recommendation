# Personalized Car Recommendation System — Streamlit App

A deployable web app version of the content-based car recommender: sidebar filters,
live-ranked results with explanations, market-position chart, and a data-exploration tab.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — sidebar filters, recommendations, charts |
| `recommender.py` | Core ML logic (cleaning, feature engineering, `PersonalizedCarRecommender`) — shared, framework-agnostic |
| `cardekho_dataset/` | Downloaded Kaggle dataset |
| `requirements.txt` | Pinned dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually `http://localhost:8501`).

## Use your own data

In the sidebar, expand **"Data source"** and upload a CSV. It must contain these columns
(rename in your CSV, or edit `DEFAULT_COLUMN_MAP` in `recommender.py` to match yours):

```
car_name, brand, selling_price, fuel_type, transmission_type,
seats, mileage, engine, max_power, km_driven, vehicle_age
```

This matches the schema of the [CarDekho used car dataset](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data).
Download it from Kaggle (requires a free account) and upload the CSV here — no code changes needed
if the column names match.

## Deploy for free — Streamlit Community Cloud

1. Push this folder (`app.py`, `recommender.py`, `requirements.txt`, `cardekho_dataset/cardekho_dataset.csv`) to a public
   (or private, on paid tiers) GitHub repo.
2. Go to **https://share.streamlit.io** → **"New app"**.
3. Point it at your repo, branch, and set the main file path to `app.py`.
4. Deploy. No secrets or environment variables are required — the dataset is bundled in `cardekho_dataset/` which means
   it works out of the box; users can still upload their own CSV once it's live.

## Deploy elsewhere (Docker-style, any cloud VM)

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Expose port `8501` (or your chosen port) through your platform's ingress/firewall rules.

## Notes

- Data cleaning, feature scaling, and model building are cached (`@st.cache_data` /
  `@st.cache_resource`), so re-running the app after the first load is fast — the pipeline only
  reruns when the input CSV changes.
- There's no user-rating ground truth in this kind of dataset, so the app doesn't report an
  "accuracy" score — see the **About** tab in the app for why, and see the companion Jupyter
  notebook for structural evaluation (coverage, diversity, filter correctness) instead.
