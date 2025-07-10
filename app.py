import pickle as pk
import pandas as pd
import streamlit as st
from pathlib import Path

path = Path.cwd().parents[0]/"Model's Save"

def model_loader(file_name):
    return pk.load(open(f'{path}/{file_name}', 'rb'))

encodeing = model_loader('Encodeing.pkl')
cluster = model_loader('Cluster.pkl')
randomforest = model_loader('RandomForest.pkl')
scaler = model_loader('Scaler.pkl')

st.title("📊 Customer Churn Prediction")

def get_user_input():

    data = {}
    vaild = ""

    for key in encodeing.keys():

        if key == 'Churn Encoding':
            continue

        feature_name = key.split()[0]

        options = encodeing[key].classes_

        user_value = st.selectbox(f"{feature_name} selection:- {options} ",['--Select Option--']+list(options))

        if user_value:

            data[feature_name] = user_value
            vaild = True
        else:
            vaild = False

    data['SeniorCitizen'] = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", ['--Select Option--',0, 1])
    data['Tenure'] = st.number_input("Tenure (months)", min_value=0)
    data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0)
    data['TotalCharges'] = data['Tenure'] * data['MonthlyCharges']
    
    return data,vaild


def encoder(df):

    for key in encodeing.keys():

        col = key.split()[0]
        try:

            df[col] = encodeing[key].transform(df[col])
        except Exception as e:
            
            print(f"Encoding error for {col}: {e}")


    return df

def preprocessing(df):

    for key in scaler.keys():

        try:
            print(key)
            col = key.split()[0]

            df[col] = scaler[key].transform(df[[col]])

        except Exception as e:

            print(f"Scaling error for {col}: {e}")

    
    return df


def cluster_group(df):

    group =  cluster.predict(df)

    if group == 2 :
       return "Moderate"
    elif group == 0:
       return 'High'
    
    else:
       return 'Low'

user_input,vaild = get_user_input()




if st.button("submit"):

    if vaild:

        data = pd.DataFrame([user_input])

        if len(data[data.isin(['--Select Option--']).any(axis=1)]) == 0:

            data.columns = [col.capitalize() for col in data.columns]
            

            data = encoder(data)
            data = preprocessing(data)

            data = (data.loc[:,randomforest.feature_names_in_])

            prob = randomforest.predict_proba(data)

            data['Churn'] = randomforest.predict(data)

            group = cluster_group(data)



            st.success(f"Churn is :- {encodeing['Churn Encoding'].inverse_transform(data['Churn'].values)[0]}")
            st.info(f'Churn Probability: {prob[0][1]*100:.2f}%')

            st.success(f'This Customer Segement :- {group} type')
        
        else:

            st.warning("⚠️ Please select all options before submitting.")

    else:
        
        st.warning("⚠️ Please provide valid values for all inputs.")

st.warning(
    "⚠️ This application predicts customer churn based on input data. "
    "However, please double-check the results, as the predictions may not always be accurate. "
    "The user is solely responsible for any decisions made based on this prediction."
)

