import os

import pandas as pd

from preprocessing import (
    CAR_INSURANCE_CATEGORY_ORDERS,
    COMPANY_SIZE_ORDER,
    FMconv,
    YNconv,
    YNlowerconv,
    alpha_categorize,
    categorize,
    categorize_TIP_ET,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_categorize_maps_known_company_sizes():
    # Ordinal by real employee-count magnitude, smallest to largest — not
    # by whatever order the categories happen to appear in the dataset.
    assert categorize('1 to 50 employees') == 0
    assert categorize('10000+ employees') == 6
    assert categorize('501 to 1000 employees') == 3


def test_categorize_is_monotonic_with_company_size():
    encoded = [categorize(size) for size in COMPANY_SIZE_ORDER]
    assert encoded == sorted(encoded)


def test_categorize_does_not_depend_on_row_order():
    # Regenerating a dataset can reshuffle which category is seen first;
    # the encoding must stay fixed regardless.
    shuffled = list(reversed(COMPANY_SIZE_ORDER))
    assert [categorize(size) for size in shuffled] == [
        COMPANY_SIZE_ORDER.index(size) for size in shuffled
    ]


def test_categorize_returns_none_for_unknown_value():
    assert categorize('not a real size') is None


def test_categorize_tip_et():
    assert categorize_TIP_ET('Government Sector') == 0
    assert categorize_TIP_ET('Private Sector/Self Employed') == 1


def test_ynconv():
    assert YNconv('Yes') == 1
    assert YNconv('No') == 0


def test_ynlowerconv():
    assert YNlowerconv('yes') == 1
    assert YNlowerconv('no') == 0


def test_fmconv():
    assert FMconv('female') == 1
    assert FMconv('male') == 0


def test_alpha_categorize_returns_index_of_first_match():
    categories = ['a', 'b', 'c']
    assert alpha_categorize('a', categories) == 0
    assert alpha_categorize('c', categories) == 2


def test_alpha_categorize_returns_none_for_unknown_value():
    assert alpha_categorize('z', ['a', 'b', 'c']) is None


def test_car_insurance_category_orders_are_semantically_ascending():
    # Each order is a real low-to-high scale, fixed independently of
    # dataset row order — pin the exact orders so a future edit that
    # reintroduces dataset-order derivation (e.g. Series.unique()) is
    # caught immediately instead of silently changing a trained model's
    # encoding.
    assert CAR_INSURANCE_CATEGORY_ORDERS == {
        'AGE': ['16-25', '26-39', '40-64', '65+'],
        'DRIVING_EXPERIENCE': ['0-9y', '10-19y', '20-29y', '30y+'],
        'EDUCATION': ['none', 'high school', 'university'],
        'INCOME': ['poverty', 'working class', 'middle class', 'upper class'],
        'VEHICLE_YEAR': ['before 2015', 'after 2015'],
    }


def test_car_insurance_category_orders_do_not_depend_on_row_order():
    for column, order in CAR_INSURANCE_CATEGORY_ORDERS.items():
        shuffled = list(reversed(order))
        encoded_from_shuffled = [alpha_categorize(v, order) for v in shuffled]
        assert encoded_from_shuffled == [order.index(v) for v in shuffled], column


def test_car_insurance_category_orders_cover_the_real_dataset():
    # Guards the explicit orders against dataset drift: if
    # Car_Insurance_Claim.csv ever gains a category we haven't accounted
    # for, train_models.py raises at training time (see
    # train_car_insurance_model) rather than silently encoding it as None.
    # This test catches the same drift without needing to train a model.
    cri = pd.read_csv(os.path.join(REPO_ROOT, 'Car_Insurance_Claim.csv'))
    for column, order in CAR_INSURANCE_CATEGORY_ORDERS.items():
        assert set(cri[column].unique()) <= set(order), column
