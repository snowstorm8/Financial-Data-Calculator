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
        'purpose_categories': sorted(loans['purpose'].unique()),
    }
    return model, metadata


def train_salary_model():
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
    cri = pd.read_csv('Car_Insurance_Claim.csv')
    cri.drop(columns=['ID', 'RACE', 'VEHICLE_TYPE', 'POSTAL_CODE'], inplace=True)
    cri['GENDER'] = cri['GENDER'].apply(FMconv)

    categorical_orders = {}
    for column in ['AGE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME', 'VEHICLE_YEAR']:
        categories = list(cri[column].unique())
        categorical_orders[column] = categories
        cri[column] = cri[column].apply(lambda x, cats=categories: alpha_categorize(x, cats))

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
        'categorical_orders': categorical_orders,
    }
    return model, metadata


def main():
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
