# Tourism Experience Analytics

## Project Title
**Tourism Experience Analytics: Classification, Prediction, and Recommendation System**

## Overview
This project analyzes tourism transaction, user, geographic, attraction, and visit-mode data to:
1. Predict attraction ratings using regression.
2. Predict visit mode using classification.
3. Recommend attractions using a content/popularity-based recommendation approach.
4. Present the results through a Streamlit application.

## Dataset
The project uses nine related datasets: Transaction, User, City, Country, Region, Continent, Type, Mode, and Updated_Item.

## Data Preparation
- Removed duplicate records.
- Checked missing values.
- Validated user, attraction, and visit-mode keys.
- Joined the normalized tables into a consolidated master dataset.
- Kept ratings in the valid 1–5 range.
- Encoded categorical features through a pipeline.

## Exploratory Data Analysis
The notebook analyzes rating distributions, visit modes, attraction types, yearly transaction trends, popular attractions, and geographic patterns.

## Machine Learning

### Regression
Target: `Rating`

Model: Random Forest Regressor

Test metrics:
- MAE: 0.7166
- RMSE: 0.9200
- R²: 0.1013

### Classification
Target: `VisitMode`

Model: Random Forest Classifier

Test metrics:
- Accuracy: 0.3818
- Weighted Precision: 0.4428
- Weighted Recall: 0.3818
- Weighted F1: 0.4017

**Important:** The classification target is not used as an input feature, avoiding direct target leakage.

## Recommendation System
The recommendation module ranks attractions using available attraction type/city content plus rating and popularity signals. The Streamlit interface supports filtering by preferred attraction type and city.

## Streamlit Application
Run locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Structure
```text
tourism-experience-analytics/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
└── notebooks/
```

## Business Insights
- Tourism businesses can use visitor-mode patterns for targeted packages and marketing.
- Attraction ratings can help identify experiences requiring service improvement.
- Attraction popularity and rating can support destination promotion.
- Personalized attraction suggestions can improve discovery and engagement.

## Limitations
- The available dataset does not contain explicit user-written reviews or sentiment text.
- Visit-mode prediction is based on the available structured demographic, time, and attraction features.
- Recommendation quality is constrained by the available attraction metadata and historical ratings.
- Model scores should be interpreted on the supplied dataset and test split rather than treated as universal performance.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter/Google Colab, Streamlit, GitHub.
