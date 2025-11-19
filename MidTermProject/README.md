# Medical Insurance Cost Prediction Model

## Problem Description
To predict medical insurance cost for a person using a person's demographics, health information

#### Where it helps:
- For insurance companies, to decide on annual premium for medical insurance
- For individuals, to get a good insurance that sufficiently covers the medical cost

#### Dataset
- Run `src/01_download_dataset.py`
- Description of data is mentioned in `src/02_EDA.ipynb`

## Exploratory Data Analysis / EDA
Code file: `src/02_EDA.ipynb`
Output files: `data/output/Correlation_Analysis(NumericFactors).xlsx`
- Factor Distribution
- Missing %
- Correlation Analysis for Numerical factors
- Target summary for categorical factors
- Feature Importance using Random Forest 

## Model Training
Code file: `src/03_model_experimentation.ipynb`
 - Baseline Model (Mean Model)
 - Linear Regression and Random Forest Regressor
 - Hyper-parameter tuning
 - Storing Model to a file
 - Prediction on an instance

## Exporting notebook to a script
Code file: `src/03.2_train.py`
Output files: `data/models/model.joblib`

## Reproducibility
- Download the data using `src/01_download_dataset.py`
- Provide the path of the downloaded dataset in `src/00_config.yaml`
- Create a pytjon enviornment using section below 'Dependency and enviroment management'
- Run `src/03.2_train.py`, that trains and stores a linear regression model at location `data/models/model.joblib`

## Model Deployment
Model can be deployed using below script
Code file: `src/04_prediction_as_service.py`

## Dependency and enviroment management
Pipefile location: `Pipfile`, `Pipfile.lock`
To install and setup the enviornment:
- cd MidTermProject
- pipenv install          # reads Pipfile and Pipfile.lock and installs all deps
- pipenv shell            # to activate the enviornment
- exit                    # to exit the enviornment

**Setup a enviornment with any pipfile**
- Change directory to the project directory
- Initialize the env using "pipenv install pandas" (if no enviorment is already initialized in the dir)
     - This will generate new enviorment name as directory name as substring
     - create Pipfile and Pipefile.lock in the same environment
     - Install pandas in the current environment
- To install other packages run above same commnad with new package name, this will install the package and update the Pipefiles 
- To activate the env in terminal
    - cd to the directory
    - run "pipenv shell"
    - if want to exit run "exit"

## Containerization
Not Done

## Cloud Deployment
Not Done
