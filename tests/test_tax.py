from tax import calculate_taxes


def test_income_below_first_bracket_is_untaxed():
    assert calculate_taxes(300000, 0) == (0.0, 0.0, 0.0)


def test_negative_taxable_income_is_untaxed():
    assert calculate_taxes(50000, 100000) == (0.0, 0.0, 0.0)


def test_income_is_taxed_marginally_not_at_flat_top_rate():
    # 977000 taxable spans four brackets (0%, 5%, 10%, 15%).
    # A flat-rate calculation would apply 15% to the whole amount and
    # return ~152412.0; the marginal calculation taxes only the slice
    # of income that falls within each bracket.
    final_tax, surcharge_amount, cess_amount = calculate_taxes(1000000, 23000)
    assert final_tax == 58812.0
    assert surcharge_amount == 0.0
    assert cess_amount == 2262.0


def test_surcharge_applies_above_five_million():
    final_tax, surcharge_amount, cess_amount = calculate_taxes(6000000, 0)
    assert surcharge_amount == 150000.0
    assert final_tax == 1716000.0
    assert cess_amount == 66000.0


def test_just_below_surcharge_threshold_has_no_surcharge():
    _, surcharge_amount, _ = calculate_taxes(4999999, 0)
    assert surcharge_amount == 0.0
