# SaaS-Churn-Prediction-and-Survival-Analysis
A Streamlit-based machine learning web application that predicts customer churn for SaaS, OTT, and EdTech platforms using engagement and demographic data. Includes churn risk scoring, survival analysis, and interactive churn prediction UI.

*Project Overview* :

A Machine Learning Web Application for Predicting Customer Churn in OTT, EdTech, and SaaS Platforms

The SaaS Churn Prediction App is an interactive Streamlit-based web dashboard that predicts customer churn for subscription-based digital platforms such as OTT streaming services, EdTech platforms, and SaaS applications.

It enables business teams to identify at-risk users early, analyze retention behavior, and develop targeted customer retention strategies based on engagement data and behavioral trends.

*Live Demo* :

Link: https://your-app-name.streamlit.app

(Hosted on Streamlit Community Cloud)

*Key Features* :

Interactive dashboard for churn prediction

Upload custom datasets (CSV format)

Visual exploratory data analysis (EDA)

Kaplan-Meier survival analysis for retention estimation

Real-time churn probability prediction

Dynamic input fields for user-specific simulation

Usable across SaaS, OTT, and EdTech domains


*Tech Stack*

Frontend/UI: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (Random Forest Classifier)

Statistical Analysis: Lifelines (Kaplan-Meier Survival Analysis)

*How to Run Locally*

*Clone the repository*

git clone https://github.com/ishu/SaaS-Churn-Prediction-App.git
cd SaaS-Churn-Prediction-App


*Install dependencies*

pip install -r requirements.txt


*Run the app*

streamlit run saas_churn_streamlit_app.py


----Open the provided local URL in a browser.----

*Model Insights* :

Users with fewer active days show higher churn probability.

Premium plan users retain longer compared to Basic plan users.

Referral-based signups show higher early churn risk.

Kaplan-Meier survival curves indicate typical retention decay over time.

*Business Impact* :

Identifies high-risk customers before they leave

Reduces customer acquisition cost through improved retention

Enables targeted marketing and save campaigns

Provides data-driven insights for product and pricing strategy
