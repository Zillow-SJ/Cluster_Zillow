# README

## Cluster Zillow Project

The Cluster Zillow Project is a data science project using the Zillow dataset to attempt to calculate log single residence homes using clustering, regression, and other ML modeling techniques.

Partners: Symeon White and Jeffery Roeder

##### Methods used:
- Applied statistics
- Machine Learning
- Data visualization

##### Technologies used:

- Python
- MySQL
- Jupyter Notebook
- Pandas


### Project needs:

The code was written in Python 3.7.3 and you will need to have pandas, numpy, matplotlib,seaborn, pandas profiling, and sklearn installed. Additionally, you will need access to the Zillow dataset which can be found at https://www.kaggle.com/c/zillow-prize-1/data

##### Files from repo:
- acquire.py
- prep.py
- explore.py
- initial_exploration.ipynb
- baseline_model.ipynb
- exploration_final.ipynb
- model_final.ipynb


### Hypothesis

Log error is being driven by lack of 'neighborhood' clusters of like houses.

### Baseline vs MVP

The baseline model used the mean of the logerror. The next model used square feet, number of bedrooms to capture estimated log error and performed slightly better (based on MSE) than the baseline..