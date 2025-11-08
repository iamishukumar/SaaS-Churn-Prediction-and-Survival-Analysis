import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter

st.set_page_config(page_title="SaaS Churn Prediction", layout="wide")


# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    # Note: You'll need to update this path or make sure the CSV is in the same folder
    # as the Streamlit app.
    try:
        df = pd.read_csv("churn_data.csv")
    except FileNotFoundError:
        st.error("Error: 'churn_data.csv' not found. Please place the data file in the same directory as the app.")
        # Create a dummy dataframe to prevent a hard crash
        data = {
            'security_no': [123, 456], 'referral_id': [11, 22], 'joining_date': ['2023-01-01', '2023-01-02'],
            'gender': ['Male', 'Female'], 'age': [30, 40], 'plan_type': ['Basic', 'Premium'],
            'active_days_90d': [10, 80], 'churn_status': ['No', 'Yes']
        }
        df = pd.DataFrame(data)
        st.warning("Using dummy data for demonstration.")

    # Clean
    df.replace({' ': np.nan}, inplace=True)
    df.dropna(inplace=True)

    # Ensure churn label
    if 'churn_status' in df.columns:
        df['churn_status'] = df['churn_status'].map({'Yes': 1, 'No': 0})
    elif 'churn_risk_score' in df.columns:
        df['churn_status'] = (df['churn_risk_score'] > 50).astype(int)

    # Ensure active_days_90d is numeric
    if 'active_days_90d' in df.columns:
        df['active_days_90d'] = pd.to_numeric(df['active_days_90d'], errors='coerce')

    # Build a list of columns to drop NaNs from that *actually exist*
    dropna_subset = []
    if 'active_days_90d' in df.columns:
        dropna_subset.append('active_days_90d')
    if 'churn_status' in df.columns:
        dropna_subset.append('churn_status')

    # Only drop NaNs if we have columns to check
    if dropna_subset:
        df.dropna(subset=dropna_subset, inplace=True)

    return df


df = load_data()

st.title("📊 SaaS Customer Churn & Engagement Prediction Dashboard")
st.markdown("End-to-end Machine Learning + Survival Analysis for SaaS churn.")

# Tabs
eda_tab, survival_tab, model_tab, predict_tab = st.tabs([
    "🔍 EDA", "📉 Subscription Lifetime", "🤖 Model Training", "🧮 Predict Churn"
])

# ------------------------------
# EDA
# ------------------------------
with eda_tab:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Detect churn column safely
    possible_churn_cols = [c for c in df.columns if 'churn' in c.lower() and 'status' in c.lower()]
    if not possible_churn_cols:
        possible_churn_cols = [c for c in df.columns if 'churn' in c.lower()]

    if possible_churn_cols:
        churn_col = possible_churn_cols[0]
    else:
        st.error("No 'churn_status' or 'churn_risk_score' column found in dataset.")
        churn_col = None

    col1, col2 = st.columns(2)

    with col1:
        if churn_col:
            st.write("### Churn Distribution")
            fig, ax = plt.subplots()
            # Ensure data is in the correct format for countplot
            if df[churn_col].dtype == 'int64' or df[churn_col].dtype == 'float64':
                sns.countplot(data=df, x=churn_col, ax=ax)
                ax.set_xticklabels(['No' if x == 0 else 'Yes' for x in df[churn_col].unique()])
            else:
                sns.countplot(data=df, x=churn_col, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Churn column not available for plotting.")

    with col2:
        st.write("### Active Days Distribution")
        if 'active_days_90d' in df.columns:
            fig2, ax2 = plt.subplots()
            sns.histplot(df['active_days_90d'], kde=True, ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("Column 'active_days_90d' not found.")

# ------------------------------
# Survival Analysis
# ------------------------------
with survival_tab:
    st.subheader("Subscription Lifetime Analysis (Survival)")

    # Use detected churn column
    possible_churn_cols = [c for c in df.columns if 'churn_status' in c.lower()]
    if not possible_churn_cols:
        possible_churn_cols = [c for c in df.columns if 'churn' in c.lower()]

    if possible_churn_cols:
        churn_col = possible_churn_cols[0]
    else:
        churn_col = None

    if churn_col and 'active_days_90d' in df.columns:
        tenure = df['active_days_90d']
        churn_event = df[churn_col]

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(tenure, event_observed=churn_event)
            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier Survival Curve (SaaS Users)")
            ax.set_xlabel("Active Days (90d)")
            ax.set_ylabel("Retention Probability")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error fitting survival curve: {e}")
    else:
        st.warning("Survival analysis skipped — missing 'churn_status' or 'active_days_90d' column.")

# ------------------------------
# Model Training
# ------------------------------
with model_tab:
    st.subheader("Train Churn Prediction Model")

    # Auto-detect churn column for model training
    possible_churn_cols = [c for c in df.columns if 'churn_status' in c.lower()]
    if not possible_churn_cols:
        possible_churn_cols = [c for c in df.columns if 'churn' in c.lower()]

    if possible_churn_cols:
        churn_col = possible_churn_cols[0]
        st.write(f"Using '{churn_col}' as the target variable.")
    else:
        st.error("No churn column found in dataset — cannot train model.")
        st.stop()

    # Drop ID-like columns and the churn column itself
    drop_cols = ['security_no', 'referral_id', 'joining_date', churn_col]
    usable_cols = [c for c in df.columns if c not in drop_cols]

    X = df[usable_cols]
    y = df[churn_col]

    # **THIS WAS THE FIX:** The following 3 lines were on one line
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    st.write(f"**Categorical features:** {', '.join(cat_cols)}")
    st.write(f"**Numerical features:** {', '.join(num_cols)}")

    preprocess = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    # Create the full model pipeline
    model = Pipeline([
        ('prep', preprocess),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        # Reduced estimators for speed
    ])

    try:
        model.fit(X, y)
        st.success("✅ Model trained successfully!")

        # Save model in session state for the prediction tab
        st.session_state['churn_model'] = model
        st.session_state['model_features'] = X.columns
        st.session_state['cat_cols'] = cat_cols
        st.session_state['num_cols'] = num_cols

    except Exception as e:
        st.error(f"Error during model training: {e}")

# ------------------------------
# Prediction Tab
# ------------------------------
with predict_tab:
    st.subheader("Predict Churn for a SaaS Customer")

    # Check if model is in session state
    if 'churn_model' in st.session_state:
        model = st.session_state['churn_model']
        X_columns = st.session_state['model_features']
        cat_cols = st.session_state['cat_cols']
        num_cols = st.session_state['num_cols']

        user_input = {}

        # Create input fields dynamically
        for col in X_columns:
            if col in cat_cols:
                options = list(df[col].unique())
                user_input[col] = st.selectbox(f"Select {col}", options=options)
            elif col in num_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                user_input[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Predict Churn"):
            inp_df = pd.DataFrame([user_input])

            # Ensure column order matches training
            inp_df = inp_df[X_columns]

            try:
                prob = model.predict_proba(inp_df)[0][1]

                # Display with a progress bar style
                st.write(f"**Churn Probability: {prob:.1%}**")
                st.progress(prob)

                if prob > 0.6:
                    st.error("⚠️ High churn risk — Customer Success should  intervene immediately.")
                elif prob > 0.4:
                    st.warning(" moderate churn risk — Consider proactive engagement.")
                else:
                    st.success("✅ Low churn risk — Customer likely to stay subscribed.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    else:
        st.warning("Model not trained yet. Please go to the '🤖 Model Training' tab and train the model first.")

st.markdown("---")
st.markdown("Built for SaaS churn prediction using ML + Survival Analysis.")