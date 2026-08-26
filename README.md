SyntHacks Savants

Overview
Our Financial Data Calculator is a tool designed to help users ascertain their eligibilty for different types of insurance policies (i.e., health, auto, travel), as well as the calculation
of income tax, loan repayability and estimated salary. By inputting specific details, the calculator provides an estimate based on predefined criteria and rules.

Features
Calculate eligibility for various types of insurance.
Calulate income tax
Check if a loan is repayable for you
Estimate an average salary with a few given inputs
User-friendly interface
Customizable parameters for accurate estimation

Insurance Premiums:
Our calculator allows you to calculate your insurance eligibility for health insurance, car insurance and travel insurance. Given a few parameters, such as age, family size, income, etc.
our calculator will be able to give you an accurate response for wheter you are eligible or not

Estimated Average Salary:
With a few inputs, we can reliably determine the average salary you can expect

Loan Repayability:
This tool, given specific data such as the installments, loan amount, credit score, etc. can reliably check whether or not a loan you take would be repayable for you or not

Income Tax:
Our tool also has the feature to calculate your income tax, based on your gross income and tax deductible.

Running it
1. Install dependencies: `pip install -r requirements.txt`
2. Train and persist the models (writes to `models/`, gitignored — you need
   to run this once before the server will start):
   `python train_models.py`
3. Start the server: `python app.py`, then open http://127.0.0.1:5000

`main.ipynb` still has the original exploratory/training cells for reference,
but `train_models.py` is the one that produces what the server actually
serves.

Current status of each feature
- Travel insurance (page + form): working, at `/` and `POST /sum`.
- Health insurance (page + form): working, at `/health` and
  `POST /calculate_health`.
- Car insurance, salary estimate, loan repayability, income tax: the models
  are trained and served as JSON APIs (`POST /calculate_car`,
  `/calculate_salary`, `/calculate_loan`, `/calculate_tax` — request bodies
  are documented in the docstrings in `app.py`), but there is no HTML form
  for them yet. The nav bar links for these are still placeholders (`#`).

Tests
`pytest` (covers the tax calculator, the categorical-encoding helpers, and
every Flask route).
