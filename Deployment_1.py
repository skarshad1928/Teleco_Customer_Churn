import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
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

elif page == "Model Training":
    st.title("Model Training - Logistic Regression")

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
        ck = numeric_for_corr.corr()
        sns.heatmap(ck, annot=True, cmap="RdBu", center=0)
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
        ct = numeric_selected.corr()
        st.write(ct)

        st.write("Heatmap After Feature Selection")
        plt.figure(figsize=(25, 25))
        sns.heatmap(ct, annot=True, cmap="RdBu", center=0)
        st.pyplot(plt.gcf())

        if "Churn Label" not in df_selected.columns:
            st.error("Churn Label not found in the selected columns. Please check the dataset.")
            st.stop()

        X = df_selected.drop("Churn Label", axis=1)
        y = df_selected["Churn Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_params = {'C': 0.01, 'penalty': None, 'solver': 'lbfgs'}
        model = LogisticRegression(
            C=best_params['C'],
            penalty=best_params['penalty'],
            solver=best_params['solver'],
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)

        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        y_pred_opt = (y_pred_proba >= best_thresh).astype(int)

        acc = accuracy_score(y_test, y_pred_opt)
        f1 = f1_score(y_test, y_pred_opt)
        auc = roc_auc_score(y_test, y_pred_proba)

        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"AUC Score: {auc:.4f}")
        st.write(f"Optimal Threshold (Youden's J): {best_thresh:.4f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_opt))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_opt))

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.scatter(fpr[ix], tpr[ix], color="red", label=f"Best Threshold = {best_thresh:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve and Optimal Threshold")
        plt.legend()
        st.pyplot(plt.gcf())

        st.subheader("Model Logistic Regression Equation (Z Equation)")
        st.markdown("""
        **Z = -1.7940 - 0.0213 × Gender + 0.0721 × Senior Citizen + 0.1387 × Partner - 0.6503 × Dependents  
        - 1.0726 × Tenure Months - 0.2304 × Phone Service + 0.3679 × Multiple Lines  
        - 0.2985 × Internet Service - 0.0778 × Online Security + 0.1565 × Online Backup  
        + 0.2707 × Device Protection - 0.0640 × Tech Support + 0.5191 × Streaming TV  
        - 0.7533 × Contract + 0.2409 × Paperless Billing + 0.0453 × Payment Method + 0.0072 × CLTV**
        """)

        joblib.dump(model, "best_logistic_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model and Scaler saved successfully.")
        
        

elif page == "Make a Prediction":
    st.title("Make a Prediction")

    try:
        model = joblib.load("best_logistic_model.pkl")
        scaler = joblib.load("scaler.pkl")
        st.write("Model loaded successfully.")
    except:
        st.error("Model not found. Train the model first in 'Model Training'.")
        st.stop()

    st.write("Enter customer details below:")

    gender_map = {'Male': 1, 'Female': 0}
    partner_map = {'Yes': 1, 'No': 0}
    dependents_map = {'Yes': 1, 'No': 0}
    phone_service_map = {'Yes': 1, 'No': 0}
    multiple_lines_map = {'Yes': 1, 'No': 0, 'No phone service': 2}
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    online_security_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    online_backup_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    device_protection_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    tech_support_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    streaming_tv_map = {'Yes': 1, 'No': 0, 'No internet service': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paperless_billing_map = {'Yes': 1, 'No': 0}
    payment_method_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }

    gender = st.selectbox("Gender", list(gender_map.keys()))
    senior_citizen = st.number_input("Senior Citizen (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
    partner = st.selectbox("Partner", list(partner_map.keys()))
    dependents = st.selectbox("Dependents", list(dependents_map.keys()))
    tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", list(phone_service_map.keys()))
    multiple_lines = st.selectbox("Multiple Lines", list(multiple_lines_map.keys()))
    internet_service = st.selectbox("Internet Service", list(internet_service_map.keys()))
    online_security = st.selectbox("Online Security", list(online_security_map.keys()))
    online_backup = st.selectbox("Online Backup", list(online_backup_map.keys()))
    device_protection = st.selectbox("Device Protection", list(device_protection_map.keys()))
    tech_support = st.selectbox("Tech Support", list(tech_support_map.keys()))
    streaming_tv = st.selectbox("Streaming TV", list(streaming_tv_map.keys()))
    contract = st.selectbox("Contract", list(contract_map.keys()))
    paperless_billing = st.selectbox("Paperless Billing", list(paperless_billing_map.keys()))
    payment_method = st.selectbox("Payment Method", list(payment_method_map.keys()))
    cltv = st.number_input("CLTV", min_value=0.0, max_value=10000.0, value=2000.0)

    inputs = {
        "Gender": gender_map[gender],
        "Senior Citizen": senior_citizen,
        "Partner": partner_map[partner],
        "Dependents": dependents_map[dependents],
        "Tenure Months": tenure,
        "Phone Service": phone_service_map[phone_service],
        "Multiple Lines": multiple_lines_map[multiple_lines],
        "Internet Service": internet_service_map[internet_service],
        "Online Security": online_security_map[online_security],
        "Online Backup": online_backup_map[online_backup],
        "Device Protection": device_protection_map[device_protection],
        "Tech Support": tech_support_map[tech_support],
        "Streaming TV": streaming_tv_map[streaming_tv],
        "Contract": contract_map[contract],
        "Paperless Billing": paperless_billing_map[paperless_billing],
        "Payment Method": payment_method_map[payment_method],
        "CLTV": cltv
    }

    if st.button("Predict Churn"):
        X_new = pd.DataFrame([inputs])
        try:
            X_new_scaled = scaler.transform(X_new)
        except Exception:
            st.error("Error scaling input features. Ensure mappings match model training.")
            st.stop()

        churn_prob = model.predict_proba(X_new_scaled)[:, 1][0]
        threshold = 0.227
        prediction = 1 if churn_prob >= threshold else 0

        st.write(f"Predicted churn probability: {churn_prob:.4f}")
        if prediction == 1:
            st.write("The customer is likely to CHURN.")
        else:
            st.write("The customer is NOT likely to churn.")

elif page == "Customer Churn Analysis":
    st.title("Customer Churn Analysis Dashboard")
    report_url = "https://app.powerbi.com/reportEmbed?reportId=2645a142-ecba-422a-9089-842afe1a29ee&autoAuth=true"
    st.components.v1.iframe(report_url, width=1200, height=800)

elif page == "Summarization":
    st.title("Summarization of Findings")

    st.markdown("""
    The Telecom Customer Churn Analysis provides a comprehensive understanding of behavioral, contractual, and service-related patterns 
    influencing customer attrition. Results show that the majority of churned users are younger, non-senior citizens, indicating a 
    higher tendency toward switching and less long-term commitment. About one-third of churned customers were single or without dependents, 
    reflecting reduced loyalty and a lower threshold for dissatisfaction. Customers using fibre optic services churned more than DSL users, 
    proving that superior infrastructure alone does not ensure satisfaction without strong customer care. Lack of technical support, 
    missing online security, and absence of reliable device protection emerged as leading churn drivers. These insights emphasize that 
    strengthening customer support, improving engagement programs for younger users, ensuring reliable network uptime, and enhancing 
    value-added services can significantly reduce churn. In summary, data-driven retention initiatives can transform reactive management 
    into proactive customer loyalty strategies for telecom firms.
    """)

    st.markdown("### Logistic Regression Equation Used for Prediction:")
    st.markdown("""
    Z = -1.7940 - 0.0213 × Gender + 0.0721 × Senior Citizen + 0.1387 × Partner - 0.6503 × Dependents  
    - 1.0726 × Tenure Months - 0.2304 × Phone Service + 0.3679 × Multiple Lines  
    - 0.2985 × Internet Service - 0.0778 × Online Security + 0.1565 × Online Backup  
    + 0.2707 × Device Protection - 0.0640 × Tech Support + 0.5191 × Streaming TV  
    - 0.7533 × Contract + 0.2409 × Paperless Billing + 0.0453 × Payment Method + 0.0072 × CLTV
    """)

    # Display Churn Summary Table
    try:
        import pandas as pd

        df = pd.read_excel("Telco_customer_churn.xlsx")
        churn_col = 'Churn Value'

        # Feature conditions for summary
        conditions = {
            'Senior Citizen == No': df['Senior Citizen'] == 'No',
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

        # Bar Chart Visualization
        st.bar_chart(churn_summary_df.set_index('Feature Condition')['Churn Rate (%)'])

        # Combined Condition Analysis
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

    except Exception as e:
        st.error(f"Error displaying churn summary table: {e}")

    st.markdown("**Resources:** [GitHub Repository](https://github.com/skarshad1928/Python/blob/main/Data_ware_House_Workspace)")
