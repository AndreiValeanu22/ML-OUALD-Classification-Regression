# ML on OUALD — Classification & Regression

**Course:** Introduction to Machine Learning (ACS, UPB)  
**Author:** [Andrei Valeanu](mailto:andrei.valeanu03@stud.acs.upb.ro)  
**Dataset:** [Open University Learning Analytics (OUALD)](https://analyse.kmi.open.ac.uk/open-dataset/help/overview)

End-to-end machine learning project on the OUALD student dataset: exploratory analysis, preprocessing, baseline and advanced models, Moodle submission, and optional **Kaggle** bonus competitions.

---

## Project overview

| Task | Target | Metric |
|------|--------|--------|
| **Classification** | inal_result (Pass / Fail / Withdrawn / Distinction) | Accuracy |
| **Regression** | inal_coursework_score (0–100) | MSE |

The notebook follows the assignment structure:

1. **3.1** — Exploratory Data Analysis (EDA)  
2. **3.2** — Preprocessing (missing values, IQR outliers, scaling, encoding)  
3. **3.3** — Models: Decision Tree, linear models, HistGradientBoosting, **CatBoost + XGBoost blend**

---

## Repository structure

`
ML-OUALD-Classification-Regression/
├── Tema1_IA_OUALD.ipynb    # Main notebook (EDA → models → export)
├── kaggle_export.py        # Moodle + Kaggle CSV generation
├── requirements.txt        # Python dependencies
├── data/
│   ├── CB_OUALD_train.csv  # Training set (18 802 rows)
│   ├── CB_OUALD_test.csv   # Moodle test set (4 700 rows)
│   └── CB_private_test.csv # Kaggle private test (4 700 rows)
└── docs/
    └── project_description.tex   # LaTeX project summary
`

---

## Quick start

### 1. Clone and install

`ash
git clone https://github.com/AndreiValeanu22/ML-OUALD-Classification-Regression.git
cd ML-OUALD-Classification-Regression
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
`

### 2. Run the notebook

`ash
jupyter notebook Tema1_IA_OUALD.ipynb
`

Run cells top to bottom. The first code cell installs dependencies via %pip install -r requirements.txt.

### 3. Generate predictions

`ash
python kaggle_export.py
`

| Output file | Purpose |
|-------------|---------|
| CB_OUALD_predictii_tema1.csv | Moodle / archive submission |
| kaggle_TEMA1_CB_clasificare_2026.csv | Kaggle classification |
| kaggle_TEMA1_CB_regresie_2026.csv | Kaggle regression |

> Full export may take **30–60 minutes** (CatBoost + XGBoost).

---

## Methods (high level)

- **Preprocessing:** median/mode imputation, IQR outlier masking, StandardScaler + one-hot for sklearn; raw DataFrames for CatBoost.
- **Baselines:** Decision Tree, Logistic Regression, Ridge / Lasso.
- **Boosting:** HistGradientBoosting, CatBoost grid search, XGBoost blend.
- **Kaggle pipeline:** multi-seed CatBoost + XGB blend + OOF regression stacking.

---

## Tech stack

Python 3.11+ · pandas · numpy · scikit-learn · CatBoost · XGBoost · matplotlib · seaborn · Jupyter

---

## LaTeX documentation

See [docs/project_description.tex](docs/project_description.tex). Compile with pdflatex docs/project_description.tex.

---

## License

MIT — see [LICENSE](LICENSE). OUALD data: [Open University terms](https://analyse.kmi.open.ac.uk/open-dataset/help/overview).
