"""
recommender.py
================
Core content-based car recommendation logic, extracted as a standalone module
so it can be imported by both the analysis notebook and the Streamlit app
without duplicating code.

Pipeline:
    raw DataFrame -> clean_data() -> build_feature_pipeline() -> PersonalizedCarRecommender
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Default schema mapping for the CarDekho used-car dataset.
# Override individual keys if your CSV's column names differ.
DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "name": "car_name",
    "brand": "brand",
    "price": "selling_price",
    "fuel_type": "fuel_type",
    "transmission": "transmission_type",
    "seats": "seats",
    "mileage": "mileage",
    "engine": "engine",
    "power": "max_power",
    "km_driven": "km_driven",
    "vehicle_age": "vehicle_age",
}


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, col_map: Dict[str, str] = DEFAULT_COLUMN_MAP) -> pd.DataFrame:
    """Drop invalid/missing rows, coerce numerics, cap price outliers, dedupe."""
    clean_df = df.copy()

    critical = [col_map[k] for k in ("price", "fuel_type", "seats") if col_map.get(k) in clean_df.columns]
    clean_df = clean_df.dropna(subset=critical)

    numeric_keys = ["price", "seats", "mileage", "engine", "power", "km_driven", "vehicle_age"]
    numeric_cols = [col_map[k] for k in numeric_keys if col_map.get(k) in clean_df.columns]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna(subset=numeric_cols)

    price_col = col_map["price"]
    clean_df = clean_df[clean_df[price_col] > 0]

    seats_col = col_map.get("seats")
    if seats_col in clean_df.columns:
        clean_df = clean_df[(clean_df[seats_col] >= 2) & (clean_df[seats_col] <= 10)]

    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    # IQR cap on price so extreme luxury listings don't dominate scaling
    q1, q3 = clean_df[price_col].quantile(0.25), clean_df[price_col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    clean_df[price_col] = clean_df[price_col].clip(lower=lower, upper=upper)

    return clean_df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_pipeline(df: pd.DataFrame, col_map: Dict[str, str] = DEFAULT_COLUMN_MAP):
    """Fit a ColumnTransformer (StandardScaler + OneHotEncoder) and return
    (feature_matrix, fitted_preprocessor, numeric_features, categorical_features)."""
    numeric_features = [col_map[k] for k in
                         ("price", "mileage", "engine", "power", "km_driven", "vehicle_age", "seats")
                         if col_map.get(k) in df.columns]
    categorical_features = [col_map[k] for k in ("fuel_type", "transmission", "brand")
                             if col_map.get(k) in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    feature_matrix = preprocessor.fit_transform(df[numeric_features + categorical_features])
    if hasattr(feature_matrix, "toarray"):
        feature_matrix = feature_matrix.toarray()

    return feature_matrix, preprocessor, numeric_features, categorical_features


# ---------------------------------------------------------------------------
# User preference schema
# ---------------------------------------------------------------------------

@dataclass
class UserPreferences:
    budget_min: float = 0
    budget_max: float = float("inf")
    preferred_fuel_types: List[str] = field(default_factory=list)
    preferred_transmission: List[str] = field(default_factory=list)
    min_seats: int = 2
    max_km_driven: float = float("inf")
    max_vehicle_age: float = float("inf")
    priorities: List[str] = field(default_factory=list)   # subset of: mileage, power, price, low_km, newer
    preferred_brands: List[str] = field(default_factory=list)
    ideal_mileage: Optional[float] = None
    ideal_power: Optional[float] = None


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class PersonalizedCarRecommender:
    PRIORITY_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

    def __init__(self, dataframe: pd.DataFrame, feature_matrix: np.ndarray,
                 preprocessor: ColumnTransformer, col_map: Dict[str, str],
                 numeric_features: list, categorical_features: list):
        self.df = dataframe.reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.preprocessor = preprocessor
        self.c = col_map
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self._ranges = {
            key: (self.df[self.c[key]].min(), self.df[self.c[key]].max())
            for key in ("price", "mileage", "power", "km_driven", "vehicle_age")
            if self.c.get(key) in self.df.columns
        }

    # ---- hard filters -----------------------------------------------
    def _hard_filter_mask(self, prefs: UserPreferences) -> pd.Series:
        c, df = self.c, self.df
        mask = (df[c["price"]] >= prefs.budget_min) & (df[c["price"]] <= prefs.budget_max)
        if prefs.preferred_fuel_types and c.get("fuel_type") in df.columns:
            mask &= df[c["fuel_type"]].isin(prefs.preferred_fuel_types)
        if prefs.preferred_transmission and c.get("transmission") in df.columns:
            mask &= df[c["transmission"]].isin(prefs.preferred_transmission)
        if c.get("seats") in df.columns:
            mask &= df[c["seats"]] >= prefs.min_seats
        if c.get("km_driven") in df.columns:
            mask &= df[c["km_driven"]] <= prefs.max_km_driven
        if c.get("vehicle_age") in df.columns:
            mask &= df[c["vehicle_age"]] <= prefs.max_vehicle_age
        return mask

    # ---- ideal-car vector ---------------------------------------------
    def _build_ideal_row(self, prefs: UserPreferences) -> pd.DataFrame:
        c = self.c
        row = {}
        if c.get("price") in self.df.columns:
            row[c["price"]] = np.clip(
                (prefs.budget_min + min(prefs.budget_max, self.df[c["price"]].max())) / 2,
                self.df[c["price"]].min(), self.df[c["price"]].max()
            )
        if c.get("mileage") in self.df.columns:
            row[c["mileage"]] = prefs.ideal_mileage or self.df[c["mileage"]].quantile(0.75)
        if c.get("power") in self.df.columns:
            row[c["power"]] = prefs.ideal_power or self.df[c["power"]].median()
        if c.get("engine") in self.df.columns:
            row[c["engine"]] = self.df[c["engine"]].median()
        if c.get("km_driven") in self.df.columns:
            row[c["km_driven"]] = min(prefs.max_km_driven, self.df[c["km_driven"]].quantile(0.25))
        if c.get("vehicle_age") in self.df.columns:
            row[c["vehicle_age"]] = min(prefs.max_vehicle_age, self.df[c["vehicle_age"]].quantile(0.25))
        if c.get("seats") in self.df.columns:
            row[c["seats"]] = max(prefs.min_seats, self.df[c["seats"]].median())
        for cat_col in self.categorical_features:
            if cat_col == c.get("fuel_type") and prefs.preferred_fuel_types:
                row[cat_col] = prefs.preferred_fuel_types[0]
            elif cat_col == c.get("transmission") and prefs.preferred_transmission:
                row[cat_col] = prefs.preferred_transmission[0]
            elif cat_col == c.get("brand") and prefs.preferred_brands:
                row[cat_col] = prefs.preferred_brands[0]
            else:
                row[cat_col] = self.df[cat_col].mode().iloc[0]
        return pd.DataFrame([row])[self.numeric_features + self.categorical_features]

    # ---- priority scoring ---------------------------------------------
    def _normalize(self, value, key, invert=False):
        lo, hi = self._ranges.get(key, (0, 1))
        if hi == lo:
            return 1.0
        norm = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        return 1.0 - norm if invert else norm

    def _priority_score(self, row, priority: str) -> float:
        c = self.c
        if priority == "mileage" and c.get("mileage") in row:
            return self._normalize(row[c["mileage"]], "mileage")
        if priority == "power" and c.get("power") in row:
            return self._normalize(row[c["power"]], "power")
        if priority == "price" and c.get("price") in row:
            return self._normalize(row[c["price"]], "price", invert=True)
        if priority == "low_km" and c.get("km_driven") in row:
            return self._normalize(row[c["km_driven"]], "km_driven", invert=True)
        if priority == "newer" and c.get("vehicle_age") in row:
            return self._normalize(row[c["vehicle_age"]], "vehicle_age", invert=True)
        return 0.0

    def _brand_bonus(self, row, prefs: UserPreferences) -> float:
        col = self.c.get("brand")
        if col and prefs.preferred_brands and row.get(col) in prefs.preferred_brands:
            return 0.05
        return 0.0

    # ---- main entry point -----------------------------------------------
    def recommend(self, prefs: UserPreferences, top_n: int = 10,
                  similarity_weight: float = 0.5) -> pd.DataFrame:
        mask = self._hard_filter_mask(prefs)
        candidate_idx = self.df[mask].index.to_numpy()
        if len(candidate_idx) == 0:
            return self.df.iloc[0:0]

        ideal_row = self._build_ideal_row(prefs)
        ideal_vector = self.preprocessor.transform(ideal_row)
        if hasattr(ideal_vector, "toarray"):
            ideal_vector = ideal_vector.toarray()

        candidate_vectors = self.feature_matrix[candidate_idx]
        sims = cosine_similarity(candidate_vectors, ideal_vector).flatten()

        priorities = prefs.priorities or ["price", "mileage", "low_km"]
        priority_scores = np.zeros(len(candidate_idx))
        for i, cidx in enumerate(candidate_idx):
            row = self.df.loc[cidx]
            p_score = sum(
                self.PRIORITY_WEIGHTS[j] * self._priority_score(row, p)
                for j, p in enumerate(priorities[:len(self.PRIORITY_WEIGHTS)])
            )
            p_score += self._brand_bonus(row, prefs)
            priority_scores[i] = p_score

        final_score = similarity_weight * sims + (1 - similarity_weight) * priority_scores

        result = self.df.loc[candidate_idx].copy()
        result["similarity_to_ideal"] = sims
        result["priority_score"] = priority_scores
        result["match_score"] = final_score
        return result.sort_values("match_score", ascending=False).head(top_n)

    def explain(self, row, prefs: UserPreferences) -> str:
        c = self.c
        reasons = []
        priorities = prefs.priorities or ["price", "mileage", "low_km"]
        for p in priorities:
            if self._priority_score(row, p) > 0.7:
                reasons.append(f"strong on {p}")
        if self._brand_bonus(row, prefs) > 0:
            reasons.append("matches your preferred brand")
        reasons.append(f"{row.get('similarity_to_ideal', 0):.0%} similar to your ideal profile")
        return ", ".join(reasons)
