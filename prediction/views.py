
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView

from .ml_model import predict_loan_uptake
from .loan_eligibility_model import predict_loan_eligibility
from .default_risk_model.default_risk_inference import predict_default_risk


# Homepage - Only accessible to authenticated users
@login_required
def home(request):
    return render(request, 'home.html')


# Login
class CustomLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = False

    def get_success_url(self):
        return '/'


# Registration
def register(request):

    if request.method == 'POST':

        form = UserCreationForm(request.POST)

        if form.is_valid():

            user = form.save(commit=False)

            user.first_name = request.POST.get('full_name', '').strip()
            user.email = request.POST.get('email', '').strip()

            user.save()

            return redirect('login')

    else:
        form = UserCreationForm()

    return render(request, 'register.html', {'form': form})


# Loan Uptake Prediction - Existing model
@login_required
def loan_prediction(request):

    result = None
    error = None

    if request.method == 'POST':

        try:
            input_data = [
                int(request.POST.get('age')),
                int(request.POST.get('experience')),
                float(request.POST.get('income')),
                int(request.POST.get('family')),
                float(request.POST.get('ccavg')),
                int(request.POST.get('education')),
                float(request.POST.get('mortgage')),
                int(request.POST.get('securities_account')),
                int(request.POST.get('cd_account')),
                int(request.POST.get('online')),
                int(request.POST.get('credit_card'))
            ]

            result = predict_loan_uptake(input_data)

        except (ValueError, TypeError):
            error = "Please enter valid values for all fields."

    if result:
        return render(request, 'loan_prediction_result.html', {
            'result': result
        })

    return render(request, 'loan_prediction.html', {
        'error': error
    })


# Loan Eligibility Prediction - Knowledge Distillation Student
@login_required
def loan_eligibility(request):

    error = None

    if request.method == 'POST':

        try:
            applicant = {
                'Age': request.POST.get('age'),
                'Annual_Income': request.POST.get('annual_income'),
                'Credit_Score': request.POST.get('credit_score'),
                'Loan_Amount': request.POST.get('loan_amount'),
                'Loan_Term_Months': request.POST.get('loan_term_months'),
                'Debt_to_Income_Ratio': request.POST.get('debt_to_income_ratio'),
                'Employment_Type': request.POST.get('employment_type'),
                'Loan_Purpose': request.POST.get('loan_purpose'),
            }

            result = predict_loan_eligibility(applicant)

            return render(request, 'loan_eligibility_result.html', {
                'result': result
            })

        except (ValueError, TypeError):
            error = "Please enter valid values for all fields."

    return render(request, 'loan_eligibility.html', {
        'error': error
    })


# Default Risk Prediction - LightGBM

DEFAULT_RISK_EMP_LENGTH_CHOICES = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
]

DEFAULT_RISK_PURPOSE_CHOICES = [
    ("car", "Car"),
    ("credit_card", "Credit Card"),
    ("debt_consolidation", "Debt Consolidation"),
    ("educational", "Educational"),
    ("home_improvement", "Home Improvement"),
    ("house", "House"),
    ("major_purchase", "Major Purchase"),
    ("medical", "Medical"),
    ("moving", "Moving"),
    ("other", "Other"),
    ("small_business", "Small Business"),
    ("vacation", "Vacation"),
    ("wedding", "Wedding"),
]


@login_required
def default_risk_prediction(request):

    error = None

    if request.method == 'POST':

        applicant = {
            'loan_amnt': request.POST.get('loan_amnt'),
            'annual_inc': request.POST.get('annual_inc'),
            'dti': request.POST.get('dti'),
            'fico_score': request.POST.get('fico_score'),

            'emp_length': request.POST.get('emp_length') or None,
            'delinq_2yrs': request.POST.get('delinq_2yrs') or None,
            'inq_last_6mths': request.POST.get('inq_last_6mths') or None,
            'open_acc': request.POST.get('open_acc') or None,
            'pub_rec': request.POST.get('pub_rec') or None,
            'revol_bal': request.POST.get('revol_bal') or None,
            'revol_util': request.POST.get('revol_util') or None,
            'total_acc': request.POST.get('total_acc') or None,
            'credit_history_months': request.POST.get('credit_history_months') or None,
            'pub_rec_bankruptcies': request.POST.get('pub_rec_bankruptcies') or None,
            'collections_12_mths_ex_med': request.POST.get('collections_12_mths_ex_med') or None,
            'tax_liens': request.POST.get('tax_liens') or None,
            'acc_now_delinq': request.POST.get('acc_now_delinq') or None,
            'delinq_amnt': request.POST.get('delinq_amnt') or None,
            'chargeoff_within_12_mths': request.POST.get('chargeoff_within_12_mths') or None,

            'home_ownership': request.POST.get('home_ownership') or None,
            'purpose': request.POST.get('purpose') or None,
        }

        try:
            result = predict_default_risk(applicant)

            probability_percent = result['default_probability'] * 100
            threshold_percent = result['decision_threshold'] * 100

            return render(request, 'default_risk_result.html', {
                'result': result,
                'probability_percent': probability_percent,
                'threshold_percent': threshold_percent,
            })

        except ValueError as exc:
            error = str(exc)

    return render(request, 'default_risk.html', {
        'error': error,
        'emp_length_choices': DEFAULT_RISK_EMP_LENGTH_CHOICES,
        'purpose_choices': DEFAULT_RISK_PURPOSE_CHOICES,
    })
