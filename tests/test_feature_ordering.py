"""Regression tests guarding against feature-order corruption.

Each ML-backed route builds a pandas DataFrame from the request body and
hands it to a persisted sklearn model. The real risk this app has already
had (see app._feature_row) is a route silently putting the right values in
the wrong columns — sklearn happily accepts a same-shaped row and predicts
on it anyway, so nothing about the response shape reveals the bug.

These tests replace each module-level model with a fake that records the
exact DataFrame it was called with, then assert both:
  - the column order matches the model's real persisted feature_columns
    (so a route can never drift from what the model was trained on), and
  - each column holds the value the request body actually meant for it
    (so a same-length reshuffle can't sneak two values past the column
    check by also swapping their positions).

This is independent of the persisted model artifacts' actual predictions,
so it stays valid across retraining.
"""

import app as app_module


class _CapturingModel:
    """Stands in for a persisted model; records what it's asked to predict."""

    def __init__(self, return_value=0):
        self.return_value = return_value
        self.calls = []

    def predict(self, row):
        self.calls.append(row)
        return [self.return_value]


def _install_fake(monkeypatch, attr_name):
    fake = _CapturingModel()
    monkeypatch.setattr(app_module, attr_name, fake)
    return fake


def test_sum_builds_row_in_travel_metadata_order(client, monkeypatch):
    fake = _install_fake(monkeypatch, '_travel_model')
    client.post('/sum', json={
        'value1': 1, 'value2': 2, 'value3': 3, 'value4': 4,
        'value5': 5, 'value6': 6, 'value7': 7, 'value8': 8,
    })
    assert len(fake.calls) == 1
    row = fake.calls[0]
    assert list(row.columns) == app_module._travel_meta['feature_columns']
    expected = {
        'Age': 1.0, 'Employment Type': 2.0, 'FamilyMembers': 3.0,
        'AnnualIncome': 4.0, 'ChronicDiseases': 5.0, 'FrequentFlyer': 6.0,
        'EverTravelledAbroad': 7.0, 'GraduateOrNot': 8.0,
    }
    assert row.iloc[0].to_dict() == expected


def test_calculate_health_builds_row_in_health_metadata_order(client, monkeypatch):
    fake = _install_fake(monkeypatch, '_health_model')
    client.post('/calculate_health', json={
        'age_health': 10, 'sex': 20, 'has_children': 30,
        'is_a_smoker': 40, 'bmi': 50,
    })
    assert len(fake.calls) == 1
    row = fake.calls[0]
    assert list(row.columns) == app_module._health_meta['feature_columns']
    expected = {'age': 10.0, 'sex': 20.0, 'bmi': 50.0, 'children': 30.0, 'smoker': 40.0}
    assert row.iloc[0].to_dict() == expected


def test_calculate_salary_builds_row_in_salary_metadata_order(client, monkeypatch):
    fake = _install_fake(monkeypatch, '_salary_model')
    client.post('/calculate_salary', json={
        'rating': 1, 'hourly': 2, 'employer_provided': 3, 'same_state': 4,
        'age': 5, 'python_yn': 6, 'r_yn': 7, 'spark': 8, 'aws': 9, 'excel': 10,
    })
    assert len(fake.calls) == 1
    row = fake.calls[0]
    assert list(row.columns) == app_module._salary_meta['feature_columns']
    expected = {
        'Rating': 1.0, 'hourly': 2.0, 'employer_provided': 3.0, 'same_state': 4.0,
        'age': 5.0, 'python_yn': 6.0, 'R_yn': 7.0, 'spark': 8.0, 'aws': 9.0, 'excel': 10.0,
    }
    assert row.iloc[0].to_dict() == expected


def test_calculate_loan_builds_row_in_loan_metadata_order(client, monkeypatch):
    fake = _install_fake(monkeypatch, '_loan_model')
    client.post('/calculate_loan', json={
        'credit.policy': 1, 'int.rate': 0.1, 'installment': 300,
        'log.annual.inc': 11, 'dti': 15, 'fico': 700,
        'days.with.cr.line': 4000, 'revol.bal': 10000, 'revol.util': 50,
        'inq.last.6mths': 1, 'delinq.2yrs': 0, 'pub.rec': 0,
        'purpose': 'educational',
    })
    assert len(fake.calls) == 1
    row = fake.calls[0]
    feature_columns = app_module._loan_meta['feature_columns']
    assert list(row.columns) == feature_columns
    values = row.iloc[0].to_dict()
    assert values['credit.policy'] == 1.0
    assert values['int.rate'] == 0.1
    assert values['fico'] == 700.0
    for column in feature_columns:
        if column.startswith('purpose_'):
            assert values[column] == (1.0 if column == 'purpose_educational' else 0.0)


def test_calculate_car_builds_row_in_car_metadata_order(client, monkeypatch):
    fake = _install_fake(monkeypatch, '_car_model')
    client.post('/calculate_car', json={
        'age': '26-39', 'gender': 'female', 'driving_experience': '10-19y',
        'education': 'university', 'income': 'middle class',
        'vehicle_year': 'after 2015', 'credit_score': 0.7,
        'vehicle_ownership': 1, 'married': 1, 'children': 0,
        'annual_mileage': 12000, 'speeding_violations': 2, 'duis': 3,
        'past_accidents': 4,
    })
    assert len(fake.calls) == 1
    row = fake.calls[0]
    feature_columns = app_module._car_meta['feature_columns']
    assert list(row.columns) == feature_columns

    orders = app_module._car_meta['categorical_orders']
    values = row.iloc[0].to_dict()
    assert values['AGE'] == float(orders['AGE'].index('26-39'))
    assert values['DRIVING_EXPERIENCE'] == float(orders['DRIVING_EXPERIENCE'].index('10-19y'))
    assert values['EDUCATION'] == float(orders['EDUCATION'].index('university'))
    assert values['INCOME'] == float(orders['INCOME'].index('middle class'))
    assert values['VEHICLE_YEAR'] == float(orders['VEHICLE_YEAR'].index('after 2015'))
    assert values['GENDER'] == 1.0  # female
    assert values['CREDIT_SCORE'] == 0.7
    assert values['ANNUAL_MILEAGE'] == 12000.0
    assert values['SPEEDING_VIOLATIONS'] == 2.0
    assert values['DUIS'] == 3.0
    assert values['PAST_ACCIDENTS'] == 4.0
