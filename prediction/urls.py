
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .forms import UsernamePasswordResetForm

urlpatterns = [
    path('', views.home, name='home'),

    path(
        'loan-prediction/',
        views.loan_prediction,
        name='loan_prediction'
    ),

    path(
        'loan-eligibility/',
        views.loan_eligibility,
        name='loan_eligibility'
    ),

    # Default Risk Prediction - LightGBM
    path(
        'default-risk/',
        views.default_risk_prediction,
        name='default_risk_prediction'
    ),

    path(
        'login/',
        views.CustomLoginView.as_view(),
        name='login'
    ),

    path(
        'register/',
        views.register,
        name='register'
    ),

    path(
        'logout/',
        auth_views.LogoutView.as_view(next_page='login'),
        name='logout'
    ),

    path(
        'forgot-password/',
        auth_views.PasswordResetView.as_view(
            template_name='password_reset.html',
            form_class=UsernamePasswordResetForm,
            email_template_name='password_reset_email.html',
            subject_template_name='password_reset_subject.txt',
            success_url='/forgot-password/sent/'
        ),
        name='password_reset'
    ),

    path(
        'forgot-password/sent/',
        auth_views.PasswordResetDoneView.as_view(
            template_name='password_reset_done.html'
        ),
        name='password_reset_done'
    ),

    path(
        'reset-password/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(
            template_name='password_reset_confirm.html',
            success_url='/reset-password/complete/'
        ),
        name='password_reset_confirm'
    ),

    path(
        'reset-password/complete/',
        auth_views.PasswordResetCompleteView.as_view(
            template_name='password_reset_complete.html'
        ),
        name='password_reset_complete'
    ),
]
