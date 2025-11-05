import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

page = st.sidebar.selectbox("Select Section", [
    "Project Overview", "Data Exploration", "Model Training",
    "Make a Prediction", "Customer Churn Analysis", "Summarization"
])

# -------------------------------
# PROJECT OVERVIEW
# -------------------------------
if page == "Project Overview":
    st.title("Customer Churn Analysis - Telecom Sector")
    st.subheader("Objective")
    st.write("The aim of this project is to predict customer churn — identifying customers who are likely to leave the telecom service.")
    
    st.subheader("Project Workflow")
    st.markdown("""
    1. Data Exploration  
    2. Data Cleaning and Preprocessing  
    3. Model Training and Evaluation  
    4. Prediction for New Customers  
    5. Business Insights  
    6. Summarization
    """)

    st.subheader("Tools and Technologies Used")
    st.markdown("""
    | Category | Tools / Libraries |
    |-----------|------------------|
    | **Programming Language** | Python |
    | **Data Handling** | Pandas, NumPy |
    | **Visualization** | Matplotlib, Seaborn, Power BI |
    | **Machine Learning** | scikit-learn (Logistic Regression) |
    | **Model Saving & Loading** | joblib |
    | **Web App Framework** | Streamlit |
    | **Data Source** | Excel (.xlsx) files |
    | **Version Control** | GitHub |
    | **Deployment** | Streamlit Cloud |
    """)

    st.write("Development Environment: Worked on Visual Studio Code with Python (Jupyter Notebook).")

    st.subheader("Project Deployment")
    st.write("""
    This project has been fully deployed using GitHub for version control and Streamlit for interactive web application deployment.  
    It demonstrates end-to-end implementation of a real-world telecom churn prediction system — from data exploration and modeling to visualization and deployment.  
    """)

    st.markdown("**Resources:** [Data_source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    st.markdown("**Resources:** [GitHub Repository](https://github.com/skarshad1928/Teleco_Customer_Churn/tree/main)")


# -------------------------------
# DATA EXPLORATION
# -------------------------------
elif page == "Data Exploration":
    st.title("Data Exploration")
    df = pd.read_excel("Telco_customer_churn.xlsx")
    st.write("First 5 Rows:")
    st.write(df.head())

    st.write("Column Names and Data Types:")
    st.write(pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Null Values": df.isnull().sum().values
    }))

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("Correlation Matrix:")
        st.write(numeric_df.corr())
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="RdBu", center=0)
        st.pyplot(plt.gcf())

    st.markdown("**Resources:** [Data_understanding_explore](https://github.com/skarshad1928/Python/blob/main/Data_ware_House_Workspace/WORK1.ipynb)")
    st.markdown("**Resources:** [Data_prep_for_model](https://github.com/skarshad1928/Python/blob/main/Data_ware_House_Workspace/WORK2.ipynb)")


# -------------------------------
# MODEL TRAINING
# -------------------------------
elif page == "Model Training":
    st.title("Model Training")

    model_choice = st.selectbox(
        "Select Model for Training",
        ["Logistic Regression", "Random Forest"]
    )

    if st.button("Train Model"):
        df = pd.read_excel("Final_Complete_cleaned.xlsx")

        id_cols = ["CustomerID", "customerID", "LID", "Unnamed: 0"]
        existing_id_cols = [c for c in id_cols if c in df.columns]
        if existing_id_cols:
            df_for_corr = df.drop(columns=existing_id_cols, errors="ignore").copy()
        else:
            df_for_corr = df.copy()

        numeric_for_corr = df_for_corr.select_dtypes(include=[np.number])
        st.subheader("Correlation Matrix Before Feature Selection (identifiers removed)")
        st.write(numeric_for_corr.corr())

        st.write("Heatmap Before Feature Selection")
        plt.figure(figsize=(25, 25))
        sns.heatmap(numeric_for_corr.corr(), annot=True, cmap="RdBu", center=0)
        st.pyplot(plt.gcf())

        selected_columns = [
            "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
            "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
            "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
            "Contract", "Paperless Billing", "Payment Method", "CLTV", "Churn Label"
        ]
        selected_existing = [c for c in selected_columns if c in df.columns]
        df_selected = df[selected_existing].copy()

        st.subheader("Correlation Matrix After Feature Selection")
        numeric_selected = df_selected.select_dtypes(include=[np.number])
        st.write(numeric_selected.corr())

        st.write("Heatmap After Feature Selection")
        plt.figure(figsize=(25, 25))
        sns.heatmap(numeric_selected.corr(), annot=True, cmap="RdBu", center=0)
        st.pyplot(plt.gcf())

        if "Churn Label" not in df_selected.columns:
            st.error("Churn Label not found in the selected columns. Please check the dataset.")
            st.stop()

        X = df_selected.drop("Churn Label", axis=1)
        y = df_selected["Churn Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train selected model
        if model_choice == "Logistic Regression":
            st.subheader("Training Logistic Regression with GridSearchCV")

            param_grid = [
                {
                    'solver': ['liblinear'],
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 0.1, 1, 10, 100]
                },
                {
                    'solver': ['lbfgs'],
                    'penalty': ['l2', None],
                    'C': [0.01, 0.1, 1, 10, 100]
                },
                {
                    'solver': ['saga'],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'l1_ratio': [0, 0.5, 1]
                }
            ]

            grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)

            best_model = grid.best_estimator_
            st.write("Best CV Score:", grid.best_score_)
            st.write("Best Parameters:", grid.best_params_)

        elif model_choice == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            st.subheader("Training Random Forest with GridSearchCV")

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            st.write("Best CV Score:", grid.best_score_)
            st.write("Best Parameters:", grid.best_params_)

        # Evaluation
        y_pred_proba = best_model.predict_proba(X_test_scaled if model_choice == "Logistic Regression" else X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        ix = np.argmax(tpr - fpr)
        best_thresh = thresholds[ix]
        y_pred_opt = (y_pred_proba >= best_thresh).astype(int)

        acc = accuracy_score(y_test, y_pred_opt)
        f1 = f1_score(y_test, y_pred_opt)
        auc = roc_auc_score(y_test, y_pred_proba)

        st.subheader("Model Evaluation on Test Set")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"AUC Score: {auc:.4f}")
        st.write(f"Optimal Threshold: {best_thresh:.4f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_opt))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_opt))

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.scatter(fpr[ix], tpr[ix], color="red", label=f"Best Threshold = {best_thresh:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())

        if model_choice == "Logistic Regression":
            joblib.dump(best_model, "best_logistic_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(best_thresh, "best_threshold.pkl")
            st.success(" Logistic Regression Model, Scaler, and Threshold saved.")

        elif model_choice == "Random Forest":
            joblib.dump(best_model, "best_random_forest_model.pkl")
            st.success(" Random Forest Model saved successfully.")


# -------------------------------
# CUSTOMER CHURN ANALYSIS (Power BI)
# -------------------------------
elif page == "Customer Churn Analysis":
    st.title("Customer Churn Analysis Dashboard")
    report_url = "https://app.powerbi.com/reportEmbed?reportId=2645a142-ecba-422a-9089-842afe1a29ee&autoAuth=true"
    st.components.v1.iframe(report_url, width=1200, height=800)


# -------------------------------
# SUMMARIZATION
# -------------------------------
elif page == "Summarization":
    st.title("Summarization of Findings")

    st.markdown("""
    The Telecom Customer Churn Analysis provides a comprehensive understanding of behavioral, contractual, and service-related patterns 
    influencing customer attrition...
    """)

    st.markdown("### Logistic Regression Equation Used for Prediction:")
    st.markdown("""
    Z = -1.7940 - 0.0213 × Gender + 0.0721 × Senior Citizen + 0.1387 × Partner - 0.6503 × Dependents  
    - 1.0726 × Tenure Months - 0.2304 × Phone Service + 0.3679 × Multiple Lines  
    - 0.2985 × Internet Service - 0.0778 × Online Security + 0.1565 × Online Backup  
    + 0.2707 × Device Protection - 0.0640 × Tech Support + 0.5191 × Streaming TV  
    - 0.7533 × Contract + 0.2409 × Paperless Billing + 0.0453 × Payment Method + 0.0072 × CLTV
    """)

    try:
        df = pd.read_excel("Telco_customer_churn.xlsx")
        churn_col = 'Churn Value'

        # Define churn-driving conditions
        conditions = {
            'Senior Citizen == Yes': df['Senior Citizen'] == 'Yes',
            'Partner == No': df['Partner'] == 'No',
            'Dependents == No': df['Dependents'] == 'No',
            'Phone Service == Yes': df['Phone Service'] == 'Yes',
            'Multiple Lines == No': df['Multiple Lines'] == 'No',
            'Internet Service == Fiber optic': df['Internet Service'] == 'Fiber optic',
            'Online Security == No': df['Online Security'] == 'No',
            'Online Backup == No': df['Online Backup'] == 'No',
            'Device Protection == No': df['Device Protection'] == 'No',
            'Tech Support == No': df['Tech Support'] == 'No',
            'Streaming TV == No': df['Streaming TV'] == 'No',
            'Contract == Month-to-month': df['Contract'] == 'Month-to-month',
            'Paperless Billing == Yes': df['Paperless Billing'] == 'Yes'
        }

        summary = []
        for label, condition in conditions.items():
            total = int(condition.sum())
            churned = int(df.loc[condition, churn_col].sum())
            churn_rate = (churned / total * 100) if total > 0 else 0
            summary.append({
                'Feature Condition': label,
                'Total Customers': total,
                'Churned Customers': churned,
                'Churn Rate (%)': round(churn_rate, 2)
            })

        churn_summary_df = pd.DataFrame(summary)
        st.markdown("### Churn Rate Summary Table (Feature-wise)")
        st.dataframe(churn_summary_df, use_container_width=True)

        sorted_df = churn_summary_df.sort_values('Churn Rate (%)', ascending=False)
        st.bar_chart(sorted_df.set_index('Feature Condition')['Churn Rate (%)'])

        # Combined Condition Summary
        combined_condition = pd.Series(True, index=df.index)
        for cond in conditions.values():
            combined_condition &= cond

        total_combined = int(combined_condition.sum())
        churned_combined = int(df.loc[combined_condition, churn_col].sum())
        churn_rate_combined = (churned_combined / total_combined * 100) if total_combined > 0 else 0

        st.markdown(f"""
        #### Combined Condition Summary (All Selected Features)
        - Total Customers: {total_combined}  
        - Churned Customers: {churned_combined}  
        - Churn Rate: {round(churn_rate_combined, 2)}%
        """)

        # --- Z-score mapping and Tenure/CLTV range logic ---
        z_mapping = {
            'Senior Citizen == Yes': 0.0721,
            'Partner == No': 0.1387,
            'Dependents == No': -0.6503,
            'Phone Service == Yes': -0.2304,
            'Multiple Lines == No': 0.3679,
            'Internet Service == Fiber optic': -0.2985,
            'Online Security == No': -0.0778,
            'Online Backup == No': 0.1565,
            'Device Protection == No': 0.2707,
            'Tech Support == No': -0.0640,
            'Streaming TV == No': 0.5191,
            'Contract == Month-to-month': -0.7533,
            'Paperless Billing == Yes': 0.2409
        }

        total_z = sum([z_mapping.get(c, 0) for c in conditions.keys()])
        st.markdown("### Logistic Z-value Analysis (Feature Conditions)")
        st.write(f"**Calculated Z value:** {total_z:.4f}")

        if total_z >= -0.875:
            st.success(f" Z = {total_z:.4f} ≥ -0.875 ⇒ Customer likely to **Churn**")
            st.markdown("""
            **Recommended Ranges (Z ≥ -0.875):**
            - **Tenure Months:** 0–15  
            - **CLTV:** below 3000  
            """)
        else:
            st.info(f" Z = {total_z:.4f} < -0.875 ⇒ Customer **not likely to churn**")
            st.markdown("""
            **Recommended Ranges (Z < -0.875):**
            - **Tenure Months:** >20  
            - **CLTV:** above 4000  
            """)

    except Exception as e:
        st.error(f"Error displaying churn summary table: {e}")

    st.markdown("**Resources:** [GitHub Repository](https://github.com/skarshad1928/Python/blob/main/Data_ware_House_Workspace)")
