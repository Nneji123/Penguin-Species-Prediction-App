![penguins](https://user-images.githubusercontent.com/101701760/171131368-98eeb20f-072f-4d22-aa66-dfcf87fd62ca.png)

# Penguin-Species-Prediction-App
A web application made with python and streamlit to predict the species of penguin given the input features.

[![Language](https://img.shields.io/badge/language-python-blue.svg?style=flat)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Streamlit-brightred.svg?style=flat)](http://www.streamlit.com)
![hosted](https://img.shields.io/badge/Streamlit-Cloud-DC143C?style=flat&logo=streamlit&logoColor=white)
![build](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat)

## Problem Statement
The goal of this project was to develop a machine learning classifier model that could accurately predict the species of penguin namely;
1. Chinstrap
2. Adelie
3. Gentoo


The predictions were based on the following features:
1. culmen_length_mm
2. culmen_depth_mm
3. flipper_length_mm
4. body_mass_g	
5. sex/gender 


## Preview
![penguin_gif](https://user-images.githubusercontent.com/101701760/171130122-b58cfdca-0e03-42f9-976e-dcabfea40077.gif)

## Data
The data used was gotten from this repository [GitHub](https://github.com/allisonhorst/penguins))

## Algorithm Used
In this project I used two different Logistic Regression classifier approaches
1. One vs One Approach:
One-vs-all classification is a method which involves training distinct binary classifiers, each designed for recognizing a particular class.


2. One vs Rest Approach:
One-vs-rest (OvR for short, also referred to as One-vs-All or OvA) is a heuristic method for using binary classification algorithms for multi-class classification. It involves splitting the multi-class dataset into multiple binary classification problems.

The final model used for the app was the Random Classifier model which had the best accuracy.


## Requirements
To run a demo do the following:
1. Clone the repository.
2. Install the requirements from the requirements.txt file:
```
pip install -r requirements.txt
```
3. Then from your command line run:
```
streamlit run penguin_predictor_app.py
```
Then you can view the site on your local server.



## Deployment
The app was made and deployed with streamlit and streamlit cloud. 
Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

The live link can be seen here:
https://share.streamlit.io/nneji123/penguin-species-prediction-app/main/penguin_predictor_app.py
