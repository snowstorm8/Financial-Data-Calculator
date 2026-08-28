# Financial Data Calculator

A small Flask web app that estimates insurance eligibility/cost, average
salary, loan repayability, and income tax from a handful of user-supplied
inputs. Predictions come from scikit-learn models trained on the sample
datasets included in this repository; the income tax calculation is plain
rule-based arithmetic (no ML involved).

This is a demo/educational project. Predictions are estimates from models
trained on the included sample datasets, not financial or insurance advice —
don't use them to make real financial, insurance, or lending decisions.

## What it does

- **Travel insurance eligibility** — predicts whether a person is likely to
  buy/qualify for travel insurance, given age, employment type, family size,
  income, and travel history.
- **Health insurance cost estimate** — predicts an expected insurance charge
  from age, sex, BMI, number of children, and smoking status.
- **Car insurance claim risk** — predicts whether a driver is likely to file
  a claim, given demographic and driving-history details.
- **Average salary estimate** — predicts an expected average salary from a
  Glassdoor-style job rating and a handful of yes/no skill and role flags.
- **Loan repayability** — predicts whether a loan is likely to be repaid,
  given financial and credit-history details plus the loan's purpose.
- **Income tax calculator** — computes tax owed, surcharge, and cess from a
  salary and deductible amount using a fixed marginal tax-bracket table (not
  a trained model).

## Key features

- Five scikit-learn models (`RandomForestClassifier` for the three
  classification tasks, `GradientBoostingRegressor` for the two regression
  tasks), trained once and persisted to disk rather than retrained per
  request.
- A metadata-driven prediction pipeline: each model's training column order
  and categorical encodings are persisted alongside it, and every route
  builds its input row by looking values up by feature name rather than
  position — eliminating a class of bug where a value could silently be fed
  to the wrong model feature.
- Explicit, semantically-ordered categorical encodings for genuinely ordinal
  fields (e.g. age bracket, income bracket, driving experience), instead of
  encodings derived from incidental dataset row order.
- A consistent JSON error contract: invalid/missing input returns `400`
  with an `{"error": ...}` body, and an unexpected server-side failure
  returns `500` with the same shape, rather than Flask's default HTML error
  page.
- Two working HTML forms (travel insurance, health insurance) with
  client-side `fetch()` calls that handle HTTP error responses and network
  failures gracefully.
- A pytest suite covering the tax calculator, the categorical-encoding
  helpers, every Flask route (success and failure paths), and dedicated
  regression tests that verify each ML route builds its feature vector in
  the correct order.

## Tech stack

- **Backend**: [Flask](https://flask.palletsprojects.com/)
- **ML / data**: [scikit-learn](https://scikit-learn.org/),
  [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/),
  [joblib](https://joblib.readthedocs.io/) (model persistence)
- **Frontend**: server-rendered Jinja2 templates with vanilla JavaScript
  (`fetch()`) and plain CSS — no frontend framework or build step
- **Testing**: [pytest](https://pytest.org/)
- **Exploration**: a Jupyter notebook (`main.ipynb`) using matplotlib/seaborn,
  kept as reference material (see [Known limitations](#known-limitations))

## Project structure

```
.
├── app.py                        # Flask app: routes, request validation, error handling
├── train_models.py               # Trains the 5 models and writes models/*.joblib
├── preprocessing.py               # Categorical/ordinal encoding helpers (shared by both)
├── tax.py                        # Rule-based income tax calculator (no ML)
├── main.ipynb                    # Original exploratory/training notebook (reference only)
├── requirements.txt               # Pinned Python dependencies
├── templates/
│   ├── base.html                 # Shared layout + nav
│   ├── main website.html         # Travel insurance form (served at "/")
│   └── health.html               # Health insurance form (served at "/health")
├── static/
│   └── style.css                 # Site-wide CSS
├── tests/
│   ├── conftest.py               # Shared Flask test-client fixture
│   ├── test_app.py                # Route-level tests (success + error paths)
│   ├── test_feature_ordering.py  # Regression tests for per-model feature ordering
│   ├── test_preprocessing.py     # Encoding-helper tests
│   └── test_tax.py                # Tax calculator tests
├── Car_Insurance_Claim.csv       # Source dataset — car insurance model
├── TravelInsurancePrediction.csv # Source dataset — travel insurance model
├── insurance.csv                  # Source dataset — health insurance model
├── loan_data.csv                  # Source dataset — loan repayability model
├── salary_data_cleaned.csv       # Source dataset — salary model
└── models/                        # Generated by train_models.py — gitignored, not in the repo
```

## Installation / setup

Requires Python 3 and pip (developed and tested with Python 3.14).

```bash
git clone <this-repository-url>
cd "<Folder Name>"
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

## How to train the models

The server loads pre-trained models from `models/`, which is gitignored and
not shipped in the repository — you must generate it once before the app
will start:

```bash
python train_models.py
```

This reads the five CSV datasets in the repo root, trains each model with a
fixed random seed (so retraining reproduces the same models), and writes
`models/<name>_model.joblib` plus `models/<name>_metadata.joblib` (feature
column order and any categorical encodings) for each of: `travel_insurance`,
`health_insurance`, `salary`, `loan`, and `car_insurance`.

## How to run the Flask app

```bash
python app.py
```

Then open http://127.0.0.1:5000. If `models/` hasn't been generated yet,
the app prints an error explaining that `train_models.py` needs to be run
first and exits, rather than starting in a broken state.

## How to run tests

```bash
pytest
```

This runs the full suite: every Flask route (valid input, missing/invalid
input, and unexpected-failure handling), the categorical-encoding helpers in
`preprocessing.py`, the tax calculator, and regression tests that check each
ML-backed route sends the correctly-ordered, correctly-labeled feature
vector to its model.

## Current feature status

- **Travel insurance** (page + form): working, at `GET /` and `POST /sum`.
- **Health insurance** (page + form): working, at `GET /health` and
  `POST /calculate_health`.
- **Car insurance, salary estimate, loan repayability, income tax**: the
  models are trained and served as JSON APIs (`POST /calculate_car`,
  `POST /calculate_salary`, `POST /calculate_loan`, `POST /calculate_tax` —
  exact request bodies are documented in each route's docstring in
  `app.py`), but there is no HTML form for any of them yet. The
  corresponding nav bar links are still placeholders (`#`).

## Known limitations

- **No HTML forms for four of the six features.** Car insurance, salary,
  loan repayability, and income tax are JSON-only APIs; the UI only covers
  travel and health insurance.
- **Model artifacts aren't committed.** `models/` is gitignored — the app
  will not start until you run `train_models.py` locally.
- **`main.ipynb` is reference material, not a build artifact.** It mirrors
  the original exploratory work but is not kept fully in sync with
  `train_models.py` (for example, it doesn't fix a random seed, and its car
  insurance section still derives categorical order from dataset row order
  rather than the fixed, semantically-ordered encoding `train_models.py`
  now uses — see the note inside the notebook). `train_models.py` is the
  authoritative training path for what the app actually serves.
- **The travel insurance form's employment-type question is inverted
  relative to how the model was trained.** The form asks the user to
  "Enter 1" for public sector / "Enter 0" for private sector, but
  `categorize_TIP_ET` in `preprocessing.py` (used when the underlying model
  was trained) encodes `Government Sector` as `0` and
  `Private Sector/Self Employed` as `1` — so a `1` submitted from the form
  is fed to the model as the opposite of what the label says.
- **No frontend automated tests.** The two inline `<script>` blocks in
  `templates/` (travel and health forms) have no JS test coverage; they've
  only been verified manually.
- **Not hardened for production.** No authentication, and `app.py` runs
  Flask's built-in development server — it isn't set up for production
  deployment as-is.
