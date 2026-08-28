"""Categorical/ordinal encoding helpers shared by train_models.py (fitting
the models) and app.py (encoding live request data the same way at
inference time). Keeping both call sites on these same functions is what
guarantees a value like 'female' or '26-39' is always turned into the same
number the model was actually trained on.
"""

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
    """Encode a company-size bracket by its position in COMPANY_SIZE_ORDER."""
    return alpha_categorize(number, COMPANY_SIZE_ORDER)


def categorize_TIP_ET(number):
    """Encode TravelInsurancePrediction.csv's 'Employment Type' column.

    Government Sector -> 0, Private Sector/Self Employed -> 1. Note this is
    the opposite of what the travel insurance form's on-page label tells the
    user to enter for "public sector" — see the known-limitation note next
    to that question in main website.html.
    """
    if number == 'Government Sector':
        return 0
    elif number == 'Private Sector/Self Employed':
        return 1


def YNconv(number):
    """Encode a 'Yes'/'No' string column (case-sensitive) as 1/0."""
    if number == 'Yes':
        return 1
    elif number == 'No':
        return 0


def YNlowerconv(number):
    """Encode a lowercase 'yes'/'no' string column as 1/0.

    Kept separate from YNconv (rather than a case-insensitive version of it)
    because it mirrors insurance.csv's actual casing exactly; the two source
    datasets happen to disagree on capitalization.
    """
    if number == 'yes':
        return 1
    elif number == 'no':
        return 0


def FMconv(number):
    """Encode gender as female -> 1, male -> 0.

    This exact mapping is baked into every model trained on a column this
    touches, so app.py's car-insurance route reproduces it by hand
    (`1.0 if gender == 'female' else 0.0`) instead of importing this
    function — keep the two in sync if this ever changes.
    """
    if number == 'female':
        return 1
    elif number == 'male':
        return 0


def alpha_categorize(number, type):
    """Return the index of `number` within the ordered list `type`.

    Used both for genuinely-ordinal encodings (COMPANY_SIZE_ORDER,
    CAR_INSURANCE_CATEGORY_ORDERS) where the index is a meaningful rank, and
    the resulting number is what the model was actually trained on.

    Silently returns None (Python's implicit function return) if `number`
    isn't found in `type` — this function assumes the caller has already
    validated membership. app.py enforces that by only ever calling this
    after `_require_category` has confirmed the value is in the allowed
    set; skipping that check would let an unrecognized category through as
    a null feature value instead of a clear error.
    """
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
