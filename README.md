# Riwaaya Analytics Dashboard

An interactive Streamlit dashboard analyzing a 1,000-response consumer survey for **Riwaaya**, a
proposed premium, authenticated Indian handicrafts marketplace. Covers descriptive analytics,
diagnostic analytics, supervised classification (membership interest prediction), regression-based
predictive analytics (spending and purchase frequency forecasting), and prescriptive business strategy.

## Project structure

```
riwaaya_dashboard/
├── app.py                          # Home page / executive overview
├── pages/
│   ├── 1_Descriptive_Analytics.py  # Demographics, buying behavior, WTP, preferences, trust, intent
│   ├── 2_Diagnostic_Analytics.py   # Key drivers, drill-downs by age/income/role/heritage/geography
│   ├── 3_Predictive_Modeling.py    # KNN, Decision Tree, Random Forest, Gradient Boosting classifiers
│   ├── 4_Predictive_Analytics.py   # Regression forecasts + conversion likelihood scoring
│   └── 5_Prescriptive_Strategy.py  # Segmentation, pricing, marketing, membership, sustainability verdict
├── riwaaya_lib/
│   ├── data_loader.py              # Single source of truth for data loading & feature engineering
│   ├── modeling.py                 # Cached classification + regression training pipelines
│   └── styling.py                  # Shared color palette and Plotly theme
├── data/
│   └── riwaaya_survey_cleaned.csv  # Cleaned survey data (1,000 rows, 56 columns)
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first visit to the Predictive Modeling and Predictive
Analytics pages will take 60-100 seconds while the models grid-search and cross-validate; results are
cached afterward via `st.cache_resource`, so it only happens once per session.

## Deploying to Streamlit Community Cloud via GitHub

1. **Create a new GitHub repository** and push this entire folder to it:
   ```bash
   git init
   git add .
   git commit -m "Riwaaya analytics dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically and launch the app.
5. The CSV in `data/` is committed to the repo, so no separate upload step is needed on the cloud side.

## Troubleshooting

**`ModuleNotFoundError` on `riwaaya_lib` (or, if you renamed it back, `utils`) after deploying:**
this almost always means the local package folder either didn't get pushed to GitHub, or the
deployed repo's folder structure doesn't match what's in this ZIP. Check, in this order:
1. On github.com, open your repo and confirm the `riwaaya_lib/` folder is actually there, with
   `__init__.py`, `data_loader.py`, `modeling.py`, and `styling.py` all inside it.
2. Confirm `app.py` sits at the **repo root**, not nested inside an extra folder, Streamlit Cloud's
   "Main file path" setting (`app.py`) is relative to the repo root.
3. If both check out and it still fails, open "Manage app" → logs on Streamlit Cloud for the full,
   unredacted traceback, it will name the exact missing module.
This project intentionally avoids the folder name `utils`, since it collides with an unrelated
package of the same name on PyPI that can occasionally get pulled in as a dependency and shadow a
same-named local folder, hence `riwaaya_lib`.

## Methodology notes

- **Classification target:** `target_membership_interest`, 1 if a respondent said "Yes, definitely" to
  joining a membership program, 0 otherwise. Chosen because it is close to perfectly balanced (about
  50.5% / 49.5%), which makes accuracy, precision, recall, and ROC-AUC all meaningfully interpretable
  without needing resampling.
- **Feature set:** ground-truth segment label (`segment_true`) is deliberately excluded from every model,
  since it would leak the answer; it isn't something Riwaaya would actually know about a real customer
  at prediction time.
- **Tuning:** all four classifiers (KNN, Decision Tree, Random Forest, Gradient Boosting) are tuned with
  `GridSearchCV` under 5-fold stratified cross-validation, optimizing ROC-AUC.
- **Regression targets:** annual spending potential and future purchase frequency are each modeled with
  their direct algebraic components excluded from the feature set, to avoid a circular, trivially "perfect"
  fit.

## Data

`data/riwaaya_survey_cleaned.csv` was produced by an upstream cleaning pipeline (duplicate removal, median/mode
imputation, ordinal and one-hot encoding, IQR-based outlier capping, and engineered business flags:
`annual_spending_potential_usd`, `high_intent_customer`, `premium_buyer`, `high_trust_customer`).
