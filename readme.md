# Loan Prediction AI

A Django-based web application featuring user registration and authentication, and machine learning-based loan predictions.

The application currently provides two prediction capabilities:

- **Loan Eligibility Prediction:** Uses LE-DKDNet (Loan Eligibility Deep Knowledge Distillation Network), a lightweight neural network trained using Knowledge Distillation, to estimate an applicant's loan eligibility.
- **Loan Uptake Prediction:** Uses the existing machine learning model to predict the likelihood of a customer taking up a loan.

## 1. Model Architecture

The application integrates two independent prediction models.

### Loan Eligibility – LE-DKDNet

LE-DKDNet (Loan Eligibility Deep Knowledge Distillation Network) is a lightweight Multi-Layer Perceptron (MLP) trained using Knowledge Distillation.

- **Teacher Model:** FLANG-BERT
- **Student Model:** Lightweight MLP neural network
- **Training Approach:** Knowledge Distillation
- **Inference:** The trained Student model performs predictions without requiring the Teacher model.
- **Framework:** PyTorch

The model estimates loan eligibility based on applicant information such as age, annual income, credit score, loan amount, loan term, debt-to-income ratio, employment type, and loan purpose.

### Loan Uptake Prediction

The application also retains its existing machine learning model for predicting the likelihood of loan uptake.

### Project Dependencies

The project dependencies include:

- **Django** – Web application framework
- **PyTorch** – Student neural network inference
- **NumPy** – Numerical operations and preprocessing
- **Scikit-learn** – Existing machine learning model and preprocessing utilities
- **python-dotenv** – Environment variable management

**Note:** PyTorch is used for the LE-DKDNet Student model. The Teacher model is not required to run the application.

## 2. Prerequisites

Install the following software before starting:

| Software | Purpose |
|---|---|
| Python 3.14 | Run the Django application |
| Git | Clone the repository |
| VS Code (recommended) | Code editor and terminal |

Download links:

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads
- VS Code: https://code.visualstudio.com/

During Python installation on Windows, make sure to select **Add Python to PATH**.

To verify the installations, open PowerShell or the VS Code terminal:

```powershell
python --version
git --version
```

## 3. Clone the Repository

Open a terminal and navigate to the directory where you want to store the project.

Clone the repository:

```powershell
git clone https://github.com/dotoeidot/loan-prediction-ai.git
```

Navigate into the project directory:

```powershell
cd loan-prediction-ai
```

Open the project in VS Code:

```powershell
code .
```

## 4. Create a Virtual Environment

A virtual environment keeps the project's Python dependencies separate from other Python projects.

Create the virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

Once activated, you should see `(venv)` at the beginning of your terminal prompt.

## 5. Install Project Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install all required Python packages:

```powershell
pip install -r requirements.txt
```

This installs Django and the other dependencies required by the application, including PyTorch, NumPy, Scikit-learn, and the environment-variable loader.

**Note:** There is no need to install Django or individual packages separately when the requirements file has been installed successfully.

## 6. Environment Configuration

The project uses a `.env` file in the root directory, alongside `manage.py`, to load the email configuration.

The `.env` file is included in the repository, so no separate environment-variable setup should be required.

Django loads these settings automatically when the application starts.

## 7. Set Up the Database

The application uses SQLite as its database.

Run the following command to apply the database migrations:

```powershell
python manage.py migrate
```

This creates the required database tables for Django authentication and the application's models.

If you want to access Django's administrative panel, create a superuser:

```powershell
python manage.py createsuperuser
```

Follow the prompts to configure the administrator username, email, and password.

This step is optional.

## 8. Start the Development Server

Run the following command:

```powershell
python manage.py runserver
```

If everything is configured correctly, Django will start the local development server.

Open your browser and visit:

**http://127.0.0.1:8000/**

You should now be able to access the Loan Prediction AI application.

To stop the server, press:

```text
CTRL + C
```

## 9. Common Commands

| Command | Purpose |
|---|---|
| `python manage.py runserver` | Start the development server |
| `python manage.py migrate` | Apply database migrations |
| `python manage.py makemigrations` | Generate migrations after model changes |
| `python manage.py createsuperuser` | Create an administrator account |
| `python manage.py check` | Check the Django project for configuration issues |
| `pip install -r requirements.txt` | Install project dependencies |

## 10. Troubleshooting

**Issue: Python is not recognized**

Python may not have been added to the Windows PATH. Reinstall Python with the PATH option enabled, or use the Python launcher:

```powershell
py --version
```

**Issue: Virtual environment activation is blocked**

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**Issue: ModuleNotFoundError**

Make sure the virtual environment is activated and run:

```powershell
pip install -r requirements.txt
```

**Issue: Database or migration errors**

Run:

```powershell
python manage.py migrate
```

**Issue: Port 8000 is already in use**

Start the application on another port:

```powershell
python manage.py runserver 8001
```

Then visit http://127.0.0.1:8001/.

---

## 11. Project Setup Summary

For future reference, once Git and Python are installed, the essential setup commands are:

```powershell
git clone https://github.com/dotoeidot/loan-prediction-ai.git

cd loan-prediction-ai

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python manage.py migrate

python manage.py runserver
```

Open http://127.0.0.1:8000/ to access the application.

The repository includes the trained model artifacts required for Loan Eligibility Prediction. No separate model training or Teacher model download is required to run predictions.

**That's it!** The application should now be running on your local development server.

## 12. Model Disclaimer

The Loan Eligibility model provides an estimated prediction based on its training data. It is intended for demonstration and research purposes and should not be interpreted as an actual loan approval or rejection decision by a financial institution.
