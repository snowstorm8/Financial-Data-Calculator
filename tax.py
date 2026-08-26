# (upper bound of bracket, rate applied to the slice of income within it)
TAX_BRACKETS = [
    (300000, 0.00),
    (600000, 0.05),
    (900000, 0.10),
    (1200000, 0.15),
    (1500000, 0.20),
    (float('inf'), 0.30),
]


def calculate_taxes(salary1, deductible):
    salary1 = float(salary1)
    deductible = float(deductible)
    cess = 0.04
    salary = salary1 - deductible

    tax_amount = 0.0
    lower_bound = 0
    for upper_bound, rate in TAX_BRACKETS:
        if salary <= lower_bound:
            break
        tax_amount += (min(salary, upper_bound) - lower_bound) * rate
        lower_bound = upper_bound

    surcharge = 0
    if salary >= 5000000:
        if salary < 10000000:
            surcharge = 0.10
        elif salary < 20000000:
            surcharge = 0.15
        elif salary >= 20000000:
            surcharge = 0.25

    surcharged_tax = tax_amount * (1 + surcharge)
    surcharge_amount = tax_amount * surcharge
    final_tax = surcharged_tax * (1 + cess)
    cess_amount = surcharged_tax * cess
    final_tax = round(final_tax, 2)
    surcharge_amount = round(surcharge_amount, 2)
    cess_amount = round(cess_amount, 2)
    return final_tax, surcharge_amount, cess_amount
