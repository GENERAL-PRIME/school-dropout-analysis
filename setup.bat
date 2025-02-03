@echo off

:: Create a virtual environment
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Install the required packages
pip install -r requirements.txt

:: Run the main program
python main.py

:: Deactivate the virtual environment (optional)
deactivate
