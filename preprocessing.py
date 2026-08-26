def categorize(number):
    if number == '501 to 1000 employees':
        return 1
    elif number == '10000+ employees':
        return 2
    elif number == '1001 to 5000 employees':
        return 3
    elif number == '51 to 200 employees':
        return 4
    elif number == '201 to 500 employees':
        return 5
    elif number == '5001 to 10000 employees':
        return 6
    elif number == '1 to 50 employees':
        return 7


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
