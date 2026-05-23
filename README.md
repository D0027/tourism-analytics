# 🌍 Tourism Experience Analytics
## Classification · Prediction · Recommendation System

**Domain:** Tourism  
**Skills:** Data Cleaning · EDA · Visualization · SQL-style Merges · Machine Learning (Regression, Classification, Recommendation) · Streamlit

---

## 📁 Project Structure

```
tourism_analytics/
├── data/                     ← Place ALL Excel dataset files here
│   ├── Transaction.xlsx
│   ├── User.xlsx
│   ├── City.xlsx
│   ├── Updated_Item.xlsx
│   ├── Mode.xlsx
│   ├── Type.xlsx
│   ├── Country.xlsx
│   ├── Region.xlsx
│   └── Continent.xlsx
│
├── models/                   ← Auto-created by train.py
│   └── tea_artifacts.pkl     ← Saved model + data artifacts
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      ← Load → Clean → Merge → Feature Engineering
│   └── models.py             ← Regression · Classification · Recommendation
│
├── app.py                    ← Streamlit dashboard (7 pages)
├── train.py                  ← One-shot training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Set up environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Add your dataset

Download the dataset from the link in the project brief and place **all Excel files** inside the `data/` folder:

```
data/
├── Transaction.xlsx
├── User.xlsx
├── City.xlsx
├── Updated_Item.xlsx      ← This replaces old Item.xlsx
├── Mode.xlsx
├── Type.xlsx
├── Country.xlsx
├── Region.xlsx
└── Continent.xlsx
```

Dataset link: https://drive.google.com/drive/folders/1U9KcwYiJ4F-Jes9-HDgxA8IaQOOKJzXt

### Step 3 — Train models

```bash
python train.py --data ./data
```

This will:
- Load and clean all datasets
- Merge them into a master DataFrame
- Engineer features (season, aggregates, encodings)
- Train 7+ regression models and select the best by RMSE
- Train 6+ classifiers and select the best by F1
- Build SVD collaborative filtering + cosine similarity content-based models
- Save everything to `models/tea_artifacts.pkl`

Training typically takes **5–15 minutes** depending on dataset size and your machine.

### Step 4 — Launch the Streamlit app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Home & Overview | KPI cards, visit mode / rating distribution, dataset summary |
| 📊 EDA & Visualizations | User demographics, attraction popularity, temporal trends, correlations |
| 🤖 Regression (Rating) | Live rating prediction, model comparison, RMSE/R² scatter |
| 🗂️ Classification (Visit Mode) | Live visit mode prediction with probability bars, classifier comparison |
| 🎯 Recommendation Engine | Collaborative filtering (SVD) + content-based (cosine similarity) |
| 📈 Model Leaderboard | Side-by-side model tables and radar chart |
| 💡 Business Insights | Key findings, actionable recommendations, next steps |

---

## 🤖 Models Trained

### Regression (predict attraction Rating 1–5)
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, Gradient Boosting, Extra Trees
- XGBoost (if installed), LightGBM (if installed)

### Classification (predict VisitMode)
- Logistic Regression, Decision Tree, Random Forest
- Gradient Boosting, Extra Trees, K-Nearest Neighbors
- XGBoost (if installed), LightGBM (if installed)

### Recommendation
- **Collaborative Filtering** — SVD on User-Item rating matrix
- **Content-Based Filtering** — Cosine similarity on attraction features

---

## 📋 Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression | RMSE, MAE, R² |
| Classification | Accuracy, F1 (weighted), Precision, Recall |
| Recommendation | RMSE, MAE (on held-out known ratings) |

---

## 🗄️ Dataset Description

| File | Purpose |
|------|---------|
| Transaction.xlsx | User visits, ratings, visit year/month, visit mode |
| User.xlsx | User demographics (continent, region, country, city) |
| City.xlsx | City names and country mapping |
| Updated_Item.xlsx | Attraction names, types, city, address |
| Mode.xlsx | Visit mode labels |
| Type.xlsx | Attraction type categories |
| Country.xlsx | Country names |
| Region.xlsx | Region names |
| Continent.xlsx | Continent names |

---

## ⚙️ Requirements

- Python 3.9+
- See `requirements.txt` for full list

Optional (install for better model performance):
```bash
pip install lightgbm xgboost
```
