from preprocessing import (
    FMconv,
    YNconv,
    YNlowerconv,
    alpha_categorize,
    categorize,
    categorize_TIP_ET,
)


def test_categorize_maps_known_company_sizes():
    assert categorize('1 to 50 employees') == 7
    assert categorize('10000+ employees') == 2


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
