"""Train the five prediction models used by app.py and persist them (plus
the encoding metadata inference needs) to models/.

This mirrors the data-cleaning steps that used to live in main.ipynb, with
one behavioral change: every split below is given a fixed random_state so
retraining does not silently change what the app predicts for the same
input. main.ipynb keeps the exploratory/training cells for reference; this
script is the one path that produces what app.py actually serves.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

from preprocessing import (
    CAR_INSURANCE_CATEGORY_ORDERS,
    FMconv,
    YNconv,
    YNlowerconv,
    alpha_categorize,
    categorize,
    categorize_TIP_ET,
)

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_loan_model():
    """Predicts `not.fully.paid` (1 = loan not repaid) from loan_data.csv.

    `purpose` is one-hot encoded with `drop_first=True`, so one category
    (alphabetically first — 'all_other') gets no column of its own and is
    instead represented as all-zeros across the other purpose_* columns.
    app.py's /calculate_loan route relies on this: it only ever sets a
    purpose_* column to 1.0 for a matching column name, so submitting
    'all_other' naturally produces the correct all-zero row without any
    special-casing.
    """
    loans = pd.read_csv('loan_data.csv')
    final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
    X = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    metadata = {
        'feature_columns': list(X.columns),
        # Every purpose value that actually exists in the training data —
        # including 'all_other' — so app.py can validate a request's
        # `purpose` against the real category set, not a hand-maintained list.
        'purpose_categories': sorted(loans['purpose'].unique()),
    }
    return model, metadata


def train_salary_model():
    """Predicts `avg_salary` (in $1000s) from salary_data_cleaned.csv.

    Free-text/high-cardinality columns (job title, description, company
    name/location, etc.) are dropped outright rather than encoded — they
    have far more categories than the models used for other features can
    productively split on, and 'Size' is dropped in favor of the ordinal
    'Categorized_Size' computed just above. `dropna(axis=1)` then removes
    any column with missing values entirely (rather than imputing rows),
    which is why the final feature set is narrower than the raw CSV.
    Note: despite the generic name, 'age' here is the company's age in
    years (derived from 'Founded'), not the applicant's age.
    """
    salary = pd.read_csv('salary_data_cleaned.csv')
    salary['Categorized_Size'] = salary['Size'].apply(categorize)
    salary.dropna(axis=1, inplace=True)
    drop_columns = [
        'Job Title', 'Salary Estimate', 'Job Description', 'Company Name',
        'Location', 'Headquarters', 'Size', 'Founded', 'Type of ownership',
        'Industry', 'Sector', 'Revenue', 'Competitors', 'min_salary',
        'max_salary', 'company_txt',
    ]
    salary.drop(columns=drop_columns, inplace=True)
    salary.drop(columns='job_state', inplace=True)
    X = salary.drop(columns='avg_salary')
    y = salary['avg_salary']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    metadata = {'feature_columns': list(X.columns)}
    return model, metadata


def train_travel_insurance_model():
    """Predicts `TravelInsurance` (1 = will buy) from
    TravelInsurancePrediction.csv. 'Unnamed: 0' is the CSV's leftover
    index column from a prior pd.to_csv() export and carries no signal.
    """
    tip = pd.read_csv('TravelInsurancePrediction.csv')
    tip['Employment Type'] = tip['Employment Type'].apply(categorize_TIP_ET)
    tip['EverTravelledAbroad'] = tip['EverTravelledAbroad'].apply(YNconv)
    tip['FrequentFlyer'] = tip['FrequentFlyer'].apply(YNconv)
    tip['GraduateOrNot'] = tip['GraduateOrNot'].apply(YNconv)
    tip.drop(columns='Unnamed: 0', inplace=True)
    X = tip.drop('TravelInsurance', axis=1)
    y = tip['TravelInsurance']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    metadata = {'feature_columns': list(X.columns)}
    return model, metadata


def train_health_insurance_model():
    """Predicts `charges` (health insurance cost) from insurance.csv.
    'region' is dropped rather than encoded — it's not exposed as an input
    on the health insurance form/API, so keeping it would make the model
    depend on a feature the app can never actually supply.
    """
    hci = pd.read_csv('insurance.csv')
    hci.drop('region', axis=1, inplace=True)
    hci['sex'] = hci['sex'].apply(FMconv)
    hci['smoker'] = hci['smoker'].apply(YNlowerconv)
    X = hci.drop('charges', axis=1)
    y = hci['charges']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    metadata = {'feature_columns': list(X.columns)}
    return model, metadata


def train_car_insurance_model():
    """Predicts `OUTCOME` (1 = claim filed) from Car_Insurance_Claim.csv.

    RACE is dropped as a fairness/legal concern rather than a modeling one
    (race should not be a factor in insurance risk pricing); ID,
    VEHICLE_TYPE, and POSTAL_CODE are dropped as non-predictive identifiers
    the app has no use for.
    """
    cri = pd.read_csv('Car_Insurance_Claim.csv')
    cri.drop(columns=['ID', 'RACE', 'VEHICLE_TYPE', 'POSTAL_CODE'], inplace=True)
    cri['GENDER'] = cri['GENDER'].apply(FMconv)

    for column, order in CAR_INSURANCE_CATEGORY_ORDERS.items():
        # Fail loudly at training time rather than silently encoding an
        # unrecognized category as None (which alpha_categorize would do,
        # producing a NaN feature the model would still happily train on).
        # A category CAR_INSURANCE_CATEGORY_ORDERS doesn't yet cover means
        # the fixed encoding in preprocessing.py needs to be updated first.
        unexpected = set(cri[column].unique()) - set(order)
        if unexpected:
            raise ValueError(
                f"{column} contains categories not covered by "
                f"CAR_INSURANCE_CATEGORY_ORDERS: {sorted(unexpected)}"
            )
        cri[column] = cri[column].apply(lambda x, cats=order: alpha_categorize(x, cats))

    # Mean imputation for the two numeric columns with missing values;
    # truncated to int (matching how this dataset's other numeric columns
    # are already whole numbers) rather than kept as a float mean.
    cri['CREDIT_SCORE'] = cri['CREDIT_SCORE'].fillna(int(np.mean(cri['CREDIT_SCORE'])))
    cri['ANNUAL_MILEAGE'] = cri['ANNUAL_MILEAGE'].fillna(int(np.mean(cri['ANNUAL_MILEAGE'])))

    X = cri.drop('OUTCOME', axis=1)
    y = cri['OUTCOME']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    metadata = {
        'feature_columns': list(X.columns),
        'categorical_orders': CAR_INSURANCE_CATEGORY_ORDERS,
    }
    return model, metadata


def main():
    """Train and persist all five models. Each model is saved alongside a
    metadata file (feature column order and any categorical encodings) so
    app.py can build correctly-ordered, correctly-encoded request rows at
    inference time without hardcoding column layout on the serving side.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainers = {
        'loan': train_loan_model,
        'salary': train_salary_model,
        'travel_insurance': train_travel_insurance_model,
        'health_insurance': train_health_insurance_model,
        'car_insurance': train_car_insurance_model,
    }
    for name, train_fn in trainers.items():
        model, metadata = train_fn()
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_model.joblib'))
        joblib.dump(metadata, os.path.join(MODELS_DIR, f'{name}_metadata.joblib'))
        print(f'Trained and saved {name}')


if __name__ == '__main__':
    main()
