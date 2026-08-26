def calculate_taxes(salary1, deductible):
    float(deductible)
    float(salary1)
    surcharge = 0
    cess = 0.04
    salary = salary1 - deductible

    # tax bracket
    if 1 == 1:
        if salary < 300000:
            tax = 0
        elif salary < 600000:
            tax = 0.05
        elif salary < 900000:
            tax = 0.10
        elif salary < 1200000:
            tax = 0.15
        elif salary < 1500000:
            tax = 0.20
        elif salary >= 1500000:
            tax = 0.30

    #surcharge
    if salary >= 5000000:
        if salary < 10000000:
            surcharge = 0.10
        elif salary < 20000000:
            surcharge = 0.15
        elif salary >= 20000000:
            surcharge = 0.25

    #finalizing tax amount
    tax_amount = salary * tax
    surcharged_tax = tax_amount * (1 + surcharge)
    surcharge_amount = tax_amount * surcharge
    final_tax = surcharged_tax * (1 + cess)
    cess_amount = surcharged_tax * cess
    final_tax = round(final_tax, 2)
    surcharge_amount = round(surcharge_amount, 2)
    cess_amount = round(cess_amount, 2)
    return final_tax, surcharge_amount, cess_amount
