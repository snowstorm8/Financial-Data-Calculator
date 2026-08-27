def test_index_renders(client):
    response = client.get('/')
    assert response.status_code == 200


def test_health_page_renders(client):
    response = client.get('/health')
    assert response.status_code == 200


def test_sum_travel_insurance(client):
    response = client.post('/sum', json={
        'value1': 30, 'value2': 1, 'value3': 3, 'value4': 500000,
        'value5': 0, 'value6': 1, 'value7': 1, 'value8': 1,
    })
    assert response.status_code == 200
    assert 'travel insurance' in response.get_json()['sum']


def test_sum_missing_body_returns_error(client):
    response = client.post('/sum', data='not json', content_type='application/json')
    assert response.status_code in (400, 401)


def test_sum_non_numeric_value_returns_400_json_not_500_html(client):
    response = client.post('/sum', json={
        'value1': 'not a number', 'value2': 1, 'value3': 1, 'value4': 1,
        'value5': 1, 'value6': 1, 'value7': 1, 'value8': 1,
    })
    assert response.status_code == 400
    assert response.content_type.startswith('application/json')
    assert 'error' in response.get_json()


def test_sum_missing_value_defaults_to_zero(client):
    # /sum's existing contract: an absent value# defaults to 0 rather than
    # erroring, unlike the other routes' _require_float fields.
    response = client.post('/sum', json={
        'value2': 1, 'value3': 1, 'value4': 1,
        'value5': 1, 'value6': 1, 'value7': 1, 'value8': 1,
    })
    assert response.status_code == 200


def test_unexpected_prediction_failure_returns_json_not_html(client, monkeypatch):
    import app as app_module

    class _BrokenModel:
        def predict(self, row):
            raise RuntimeError('simulated model/artifact failure')

    monkeypatch.setattr(app_module, '_health_model', _BrokenModel())
    response = client.post('/calculate_health', json={
        'age_health': 20, 'sex': 1, 'has_children': 1,
        'is_a_smoker': 1, 'bmi': 20.5,
    })
    assert response.status_code == 500
    assert response.content_type.startswith('application/json')
    assert 'error' in response.get_json()


def test_calculate_health_valid_input(client):
    response = client.post('/calculate_health', json={
        'age_health': 20, 'sex': 1, 'has_children': 1,
        'is_a_smoker': 1, 'bmi': 20.5,
    })
    assert response.status_code == 200
    assert 'health_sum' in response.get_json()


def test_calculate_health_missing_field_returns_400(client):
    response = client.post('/calculate_health', json={'age_health': 20})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_calculate_salary_valid_input(client):
    response = client.post('/calculate_salary', json={
        'rating': 4.5, 'hourly': 0, 'employer_provided': 0,
        'same_state': 1, 'age': 25, 'python_yn': 1, 'r_yn': 0,
        'spark': 0, 'aws': 1, 'excel': 1,
    })
    assert response.status_code == 200
    assert 'predicted_salary_k' in response.get_json()


def test_calculate_salary_missing_field_returns_400(client):
    response = client.post('/calculate_salary', json={'rating': 4.5})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_calculate_salary_invalid_type_returns_400(client):
    response = client.post('/calculate_salary', json={
        'rating': 'not a number', 'hourly': 0, 'employer_provided': 0,
        'same_state': 1, 'age': 25, 'python_yn': 1, 'r_yn': 0,
        'spark': 0, 'aws': 1, 'excel': 1,
    })
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_calculate_loan_valid_input(client):
    response = client.post('/calculate_loan', json={
        'credit.policy': 1, 'int.rate': 0.1, 'installment': 300,
        'log.annual.inc': 11, 'dti': 15, 'fico': 700,
        'days.with.cr.line': 4000, 'revol.bal': 10000, 'revol.util': 50,
        'inq.last.6mths': 1, 'delinq.2yrs': 0, 'pub.rec': 0,
        'purpose': 'debt_consolidation',
    })
    assert response.status_code == 200
    assert isinstance(response.get_json()['loan_repayable'], bool)


def test_calculate_loan_unknown_purpose_returns_400(client):
    response = client.post('/calculate_loan', json={
        'credit.policy': 1, 'int.rate': 0.1, 'installment': 300,
        'log.annual.inc': 11, 'dti': 15, 'fico': 700,
        'days.with.cr.line': 4000, 'revol.bal': 10000, 'revol.util': 50,
        'inq.last.6mths': 1, 'delinq.2yrs': 0, 'pub.rec': 0,
        'purpose': 'not_a_real_purpose',
    })
    assert response.status_code == 400


def test_calculate_car_valid_input(client):
    response = client.post('/calculate_car', json={
        'age': '26-39', 'gender': 'female', 'driving_experience': '10-19y',
        'education': 'university', 'income': 'middle class',
        'vehicle_year': 'after 2015', 'credit_score': 0.7,
        'vehicle_ownership': 1, 'married': 1, 'children': 0,
        'annual_mileage': 12000, 'speeding_violations': 0, 'duis': 0,
        'past_accidents': 0,
    })
    assert response.status_code == 200
    assert isinstance(response.get_json()['claim_predicted'], bool)


def test_calculate_car_unknown_category_returns_400(client):
    response = client.post('/calculate_car', json={
        'age': 'not a real age bracket', 'gender': 'female',
        'driving_experience': '10-19y', 'education': 'university',
        'income': 'middle class', 'vehicle_year': 'after 2015',
        'credit_score': 0.7, 'vehicle_ownership': 1, 'married': 1,
        'children': 0, 'annual_mileage': 12000, 'speeding_violations': 0,
        'duis': 0, 'past_accidents': 0,
    })
    assert response.status_code == 400


def test_calculate_tax_valid_input(client):
    response = client.post('/calculate_tax', json={'salary': 1000000, 'deductible': 23000})
    assert response.status_code == 200
    body = response.get_json()
    assert body['final_tax'] == 58812.0


def test_calculate_tax_missing_field_returns_400(client):
    response = client.post('/calculate_tax', json={'salary': 1000000})
    assert response.status_code == 400
