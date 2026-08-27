"""Flask server for the financial calculator.

Loads models trained and persisted by train_models.py (run that first —
see README) instead of retraining on every request. Each route's JSON
contract is documented in its docstring.
"""

import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from preprocessing import alpha_categorize
from tax import calculate_taxes

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _load(name):
    model_path = os.path.join(MODELS_DIR, f'{name}_model.joblib')
    metadata_path = os.path.join(MODELS_DIR, f'{name}_metadata.joblib')
    if not os.path.exists(model_path):
        sys.exit(
            f"Missing {model_path}. Run 'python train_models.py' first to "
            "train and persist the models this app serves."
        )
    return joblib.load(model_path), joblib.load(metadata_path)


_travel_model, _travel_meta = _load('travel_insurance')
_health_model, _health_meta = _load('health_insurance')
_salary_model, _salary_meta = _load('salary')
_loan_model, _loan_meta = _load('loan')
_car_model, _car_meta = _load('car_insurance')


class InvalidInput(ValueError):
    pass


def _require_float(data, key):
    if key not in data:
        raise InvalidInput(f"missing field '{key}'")
    try:
        return float(data[key])
    except (TypeError, ValueError):
        raise InvalidInput(f"field '{key}' must be a number")


def _require_category(data, key, allowed):
    if key not in data:
        raise InvalidInput(f"missing field '{key}'")
    value = data[key]
    if value not in allowed:
        raise InvalidInput(f"field '{key}' must be one of {sorted(allowed)}")
    return value


def _optional_float(data, key, default=0):
    """Like _require_float, but a missing key falls back to `default`
    instead of a 400 — this is /sum's existing, intentionally lenient
    contract. A *present* non-numeric value is still a 400, not a crash."""
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        raise InvalidInput(f"field '{key}' must be a number")


def _feature_row(values, feature_columns):
    """Build a single-row DataFrame ordered by the model's actual training
    columns (persisted in metadata), keyed by name rather than position.

    A plain positional list silently mislabels a feature if the metadata's
    column order ever changes (retraining, dataset edits); looking each
    value up by name removes that failure mode. A KeyError here means
    `values` doesn't actually cover the model's features, which is a
    programming error and should surface, not be swallowed.
    """
    return pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns,
    )


@app.errorhandler(InvalidInput)
def _handle_invalid_input(error):
    return jsonify({'error': str(error)}), 400


@app.errorhandler(Exception)
def _handle_unexpected_error(error):
    """Every route here is a JSON API (aside from the two HTML pages), so an
    unhandled error — a prediction-time crash, a corrupt model artifact,
    a genuine bug — should still come back as JSON, not Flask's default
    HTML 500 page. HTTPExceptions (404, 405, ...) are left to their normal
    handling; only genuinely unexpected exceptions are converted here, and
    they're logged in full rather than silently swallowed.
    """
    if isinstance(error, HTTPException):
        return error
    app.logger.exception('Unhandled exception on %s', request.path)
    return jsonify({'error': 'internal server error'}), 500


@app.route('/')
def index():
    return render_template('main website.html')


@app.route('/health')
def health_page():
    return render_template('health.html')


@app.route('/sum', methods=['POST'])
def sum():
    """Travel insurance eligibility. Existing contract, unchanged:
    body {value1..value8} -> {"sum": "<result text>"}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401
    val1 = _optional_float(data, 'value1')
    val2 = _optional_float(data, 'value2')
    val3 = _optional_float(data, 'value3')
    val4 = _optional_float(data, 'value4')
    val5 = _optional_float(data, 'value5')
    val6 = _optional_float(data, 'value6')
    val7 = _optional_float(data, 'value7')
    val8 = _optional_float(data, 'value8')
    values = {
        'Age': val1,
        'Employment Type': val2,
        'GraduateOrNot': val8,
        'AnnualIncome': val4,
        'FamilyMembers': val3,
        'ChronicDiseases': val5,
        'FrequentFlyer': val6,
        'EverTravelledAbroad': val7,
    }
    row = _feature_row(values, _travel_meta['feature_columns'])
    if _travel_model.predict(row)[0] == 1:
        sum = 'You will get travel insurance'
    else:
        sum = 'You will not get travel insurance'
    return jsonify({'sum': sum})


@app.route('/calculate_health', methods=['POST'])
def calculate_health():
    """Health insurance cost estimate. Matches health.html's existing
    fetch() contract: body {age_health, sex, has_children, is_a_smoker,
    bmi} -> {"health_sum": "<result text>"}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401
    age = _require_float(data, 'age_health')
    sex = _require_float(data, 'sex')
    has_children = _require_float(data, 'has_children')
    is_a_smoker = _require_float(data, 'is_a_smoker')
    bmi = _require_float(data, 'bmi')
    values = {'age': age, 'sex': sex, 'bmi': bmi, 'children': has_children, 'smoker': is_a_smoker}
    row = _feature_row(values, _health_meta['feature_columns'])
    predicted_cost = _health_model.predict(row)[0]
    return jsonify({'health_sum': f'Estimated health insurance cost: ${predicted_cost:,.2f}'})


@app.route('/calculate_salary', methods=['POST'])
def calculate_salary():
    """Estimated average salary (in thousands of dollars). Body must
    contain: rating, hourly, employer_provided, same_state, age,
    python_yn, r_yn, spark, aws, excel -> {"predicted_salary_k": <float>}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401
    values = {
        'Rating': _require_float(data, 'rating'),
        'hourly': _require_float(data, 'hourly'),
        'employer_provided': _require_float(data, 'employer_provided'),
        'same_state': _require_float(data, 'same_state'),
        'age': _require_float(data, 'age'),
        'python_yn': _require_float(data, 'python_yn'),
        'R_yn': _require_float(data, 'r_yn'),
        'spark': _require_float(data, 'spark'),
        'aws': _require_float(data, 'aws'),
        'excel': _require_float(data, 'excel'),
    }
    row = _feature_row(values, _salary_meta['feature_columns'])
    predicted = _salary_model.predict(row)[0]
    return jsonify({'predicted_salary_k': round(float(predicted), 2)})


@app.route('/calculate_loan', methods=['POST'])
def calculate_loan():
    """Loan repayability. Body must contain the twelve numeric loan
    fields (credit.policy, int.rate, installment, log.annual.inc, dti,
    fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths,
    delinq.2yrs, pub.rec) plus purpose (one of the known purpose
    categories) -> {"loan_repayable": <bool>}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401

    numeric_fields = [c for c in _loan_meta['feature_columns'] if not c.startswith('purpose_')]
    values = {field: _require_float(data, field) for field in numeric_fields}
    purpose = _require_category(data, 'purpose', set(_loan_meta['purpose_categories']))
    for column in _loan_meta['feature_columns']:
        if column.startswith('purpose_'):
            values[column] = 1.0 if column == f'purpose_{purpose}' else 0.0

    row_df = _feature_row(values, _loan_meta['feature_columns'])
    prediction = _loan_model.predict(row_df)[0]
    return jsonify({'loan_repayable': bool(prediction == 0)})


@app.route('/calculate_car', methods=['POST'])
def calculate_car():
    """Car insurance claim risk. Body must contain: age (one of '16-25',
    '26-39', '40-64', '65+'), gender ('male'/'female'), driving_experience
    (one of '0-9y', '10-19y', '20-29y', '30y+'), education (one of
    'high school', 'university', 'none'), income (one of 'poverty',
    'working class', 'middle class', 'upper class'), vehicle_year
    ('before 2015'/'after 2015'), credit_score, vehicle_ownership, married,
    children, annual_mileage, speeding_violations, duis, past_accidents
    -> {"claim_predicted": <bool>}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401

    orders = _car_meta['categorical_orders']
    age = alpha_categorize(_require_category(data, 'age', set(orders['AGE'])), orders['AGE'])
    driving_experience = alpha_categorize(
        _require_category(data, 'driving_experience', set(orders['DRIVING_EXPERIENCE'])),
        orders['DRIVING_EXPERIENCE'],
    )
    education = alpha_categorize(
        _require_category(data, 'education', set(orders['EDUCATION'])), orders['EDUCATION']
    )
    income = alpha_categorize(
        _require_category(data, 'income', set(orders['INCOME'])), orders['INCOME']
    )
    vehicle_year = alpha_categorize(
        _require_category(data, 'vehicle_year', set(orders['VEHICLE_YEAR'])),
        orders['VEHICLE_YEAR'],
    )
    gender = _require_category(data, 'gender', {'male', 'female'})
    gender_value = 1.0 if gender == 'female' else 0.0

    row = {
        'AGE': age,
        'GENDER': gender_value,
        'DRIVING_EXPERIENCE': driving_experience,
        'EDUCATION': education,
        'INCOME': income,
        'CREDIT_SCORE': _require_float(data, 'credit_score'),
        'VEHICLE_OWNERSHIP': _require_float(data, 'vehicle_ownership'),
        'VEHICLE_YEAR': vehicle_year,
        'MARRIED': _require_float(data, 'married'),
        'CHILDREN': _require_float(data, 'children'),
        'ANNUAL_MILEAGE': _require_float(data, 'annual_mileage'),
        'SPEEDING_VIOLATIONS': _require_float(data, 'speeding_violations'),
        'DUIS': _require_float(data, 'duis'),
        'PAST_ACCIDENTS': _require_float(data, 'past_accidents'),
    }
    row_df = _feature_row(row, _car_meta['feature_columns'])
    prediction = _car_model.predict(row_df)[0]
    return jsonify({'claim_predicted': bool(prediction == 1)})


@app.route('/calculate_tax', methods=['POST'])
def calculate_tax():
    """Income tax. Body {salary, deductible} ->
    {"final_tax", "surcharge_amount", "cess_amount"}."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 401
    salary = _require_float(data, 'salary')
    deductible = _require_float(data, 'deductible')
    final_tax, surcharge_amount, cess_amount = calculate_taxes(salary, deductible)
    return jsonify({
        'final_tax': final_tax,
        'surcharge_amount': surcharge_amount,
        'cess_amount': cess_amount,
    })


if __name__ == '__main__':
    app.run()
