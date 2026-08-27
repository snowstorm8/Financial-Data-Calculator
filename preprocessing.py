# Genuinely ordinal (company size, smallest to largest), so the encoding is
# a fixed, explicit order rather than derived from row/appearance order in
# whatever dataset happens to be loaded.
COMPANY_SIZE_ORDER = [
    '1 to 50 employees',
    '51 to 200 employees',
    '201 to 500 employees',
    '501 to 1000 employees',
    '1001 to 5000 employees',
    '5001 to 10000 employees',
    '10000+ employees',
]


def categorize(number):
    return alpha_categorize(number, COMPANY_SIZE_ORDER)


def categorize_TIP_ET(number):
    if number == 'Government Sector':
        return 0
    elif number == 'Private Sector/Self Employed':
        return 1


def YNconv(number):
    if number == 'Yes':
        return 1
    elif number == 'No':
        return 0


def YNlowerconv(number):
    if number == 'yes':
        return 1
    elif number == 'no':
        return 0


def FMconv(number):
    if number == 'female':
        return 1
    elif number == 'male':
        return 0


def alpha_categorize(number, type):
    for num in range(len(type)):
        if number == type[num]:
            return num


# Explicit, semantically meaningful orderings for Car_Insurance_Claim.csv's
# genuinely ordinal columns (each is a real low-to-high scale: age, driving
# experience, education level, income bracket, and vehicle age). Fixed here
# instead of derived from `Series.unique()`, whose order reflects incidental
# row order in the CSV and would silently change the numeric encoding if the
# dataset were ever regenerated, reshuffled, or resampled.
CAR_INSURANCE_CATEGORY_ORDERS = {
    'AGE': ['16-25', '26-39', '40-64', '65+'],
    'DRIVING_EXPERIENCE': ['0-9y', '10-19y', '20-29y', '30y+'],
    'EDUCATION': ['none', 'high school', 'university'],
    'INCOME': ['poverty', 'working class', 'middle class', 'upper class'],
    'VEHICLE_YEAR': ['before 2015', 'after 2015'],
}
