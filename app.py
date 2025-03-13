import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the data
df1 = pd.read_csv(r"C:\Users\prans\OneDrive\Desktop\OLD PC\Desktop\house pridiction\Bengaluru_House_Data.csv")

# Data preprocessing functions
def preprocess_data(df):
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

    # Extract numeric values from 'size'
    df2['bhk'] = df2['size'].str.extract(r'(\d+)', expand=False)  # ✅ Fixed escape sequence
    df2['bhk'] = pd.to_numeric(df2['bhk'], errors='coerce')  # Convert to numeric, coerce NaN values
    df2.dropna(subset=['bhk'], inplace=True)  # Remove rows with NaN in 'bhk' column

    # Convert sqft to numeric format
    df2['total_sqft'] = df2['total_sqft'].apply(convert_sqft_to_num)
    df2 = df2[df2.total_sqft.notnull()]  # Remove rows with null sqft values
    
    # Create price_per_sqft feature
    df2['price_per_sqft'] = df2['price'] * 100000 / df2['total_sqft']
    
    # Handle locations with less than 10 occurrences
    location_stats = df2['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df2['location'] = df2['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    return df2

def convert_sqft_to_num(x):
    try:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

# Train the model
def train_model(df):
    df_processed = preprocess_data(df)
    
    # Ensure 'price' exists before dropping
    if 'price' not in df_processed.columns:
        raise KeyError("Column 'price' not found in dataset after preprocessing.")

    X = df_processed.drop(['price', 'location', 'size', 'price_per_sqft'], axis='columns', errors='ignore')
    y = df_processed['price']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=10)

    # Train model
    lr_clf = LinearRegression()
    lr_clf.fit(X_train, y_train)

    return {'model': lr_clf, 'imputer': imputer, 'columns': X.columns}

# Prediction function
def predict_price(model_info, location, sqft, bath, bhk):
    model = model_info['model']
    columns = model_info['columns']

    x = np.zeros(len(columns))
    if 'total_sqft' in columns:
        x[np.where(columns == 'total_sqft')[0][0]] = sqft
    if 'bath' in columns:
        x[np.where(columns == 'bath')[0][0]] = bath
    if 'bhk' in columns:
        x[np.where(columns == 'bhk')[0][0]] = bhk

    # Handle categorical 'location' (not included in model)
    return model.predict([x])[0]

# Train model
trained_model = train_model(df1)
model_info = {'model': trained_model['model'], 'columns': trained_model['columns']}

# Streamlit UI
st.title('House Price Prediction')
st.sidebar.header('Input Parameters')

# Input fields
location = st.sidebar.selectbox('Location', df1['location'].unique())
sqft = st.sidebar.number_input('Square Feet', min_value=0.0, step=1.0)
bath = st.sidebar.number_input('Bathrooms', min_value=0, step=1)
bhk = st.sidebar.number_input('Bedrooms', min_value=0, step=1)

if st.sidebar.button('Predict'):
    prediction = predict_price(model_info, location, sqft, bath, bhk)
    st.write(f"Estimated Price: {prediction:.2f} Lakhs")
