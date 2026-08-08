# Personalized Car Recommendation System

This notebook implements a **Personalized Car Recommendation System**, a content-based machine learning recommendation engine that operates on real-world used-car data.

## Dataset

The project utilizes the [CarDekho Used Car Data](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data) dataset from Kaggle.

## Project Goals

1.  **Data Exploration and Cleaning:** Thoroughly explore and clean a real-world used-car listings dataset.
2.  **Feature Engineering:** Develop features suitable for similarity-based recommendation.
3.  **Content-Based Recommender:** Construct a content-based recommender using **cosine similarity** across a feature space combining numeric and categorical data.
4.  **Hybrid System:** Integrate an explicit **user-preference scoring layer** (a hybrid approach blending similarity with rule-based filters).
5.  **Evaluation:** Assess the system's performance (coverage, diversity, sanity checks) without direct user-rating ground truth.
6.  **Interactive Demo:** Provide an interactive function that allows users to query the system for recommendations.

## Notebook Structure

The notebook is organized into the following sections:

1.  **Setup & Data Loading:** Initial setup and loading of the dataset.
2.  **Exploratory Data Analysis (EDA):** Understanding data distributions, missing values, and relationships.
3.  **Data Cleaning & Preprocessing:** Handling missing values, coering data types, and outlier capping.
4.  **Feature Engineering:** Creating numeric features and one-hot encoding categorical features.
5.  **Model: Content-Based Recommender:** Implementing the core similarity model using `NearestNeighbors` and cosine similarity.
6.  **Hybrid Layer:** Developing the personalized filtering and re-ranking logic based on user preferences.
7.  **Evaluation:** Assessing the system's performance using structural metrics.
8.  **Interactive Demo:** Providing a user-friendly interface to test the recommender.
9.  **Conclusion & Next Steps:** Summarizing the system's capabilities, limitations, and future enhancements.
