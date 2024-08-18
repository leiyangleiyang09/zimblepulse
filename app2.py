#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Load your trained clustering model
with open('clustering_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

# Define the columns that were removed during training due to high missing values
columns_removed_during_training = [
    'Phone', 'eMail', 'Comment systems', 'Cookie compliance', 'Accessibility', 
    'Appointment scheduling', 'Surveys', 'Geolocation', 'Browser fingerprinting', 
    'Segmentation', 'Translation', 'Referral marketing', 'Digital asset management', 
    'Content curation', 'Cart abandonment', 'Shipping carriers', 
    'Recruitment & staffing', 'Returns', 'Mobile frameworks', 'User onboarding', 
    'IaaS', 'Form builders', 'Caching', 'Reverse proxies', 'Load balancers', 
    'WordPress themes', 'JavaScript graphics', 'Operating systems', 
    'Cross border ecommerce', 'Fulfilment', 'Ecommerce frontends', 
    'Rich text editors', 'Editor', 'Documentation tools', 'Hosting panels', 
    'CMS', 'Photo galleries', 'Blogs', 'LMS', 'Page builder', 
    'Static site generators'
]

# Define the expected columns based on the training data
expected_columns = [
    'Broad Vertical_Others', 'Broad Vertical_Retail', 'Broad Vertical_Services', 
    'Tech Adoption Level_Low', 'Tech Adoption Level_Medium', 'Tech Adoption Level_High'
]

# Cluster insights dictionary
cluster_insights = {
    0: {
        "name": "High Tech Usage",
        "profile": "Advanced tech users, predominantly from Retail and Manufacturing sectors.",
        "key_technologies": [
            "CDN Technologies: High adoption of Cloudflare (27.82%) and Amazon S3 (3.75%).",
            "Analytics Tools: Significant usage of Facebook Pixel (33.08%) and Google Analytics (12.78%).",
            "Security Technologies: Notable adoption of HSTS (24.06%).",
            "JavaScript Libraries: High usage of core-js (21.05%) and Lodash (13.53%)."
        ],
        "insights": [
            "These companies are leveraging advanced technologies to enhance performance, security, and analytics.",
            "They are likely early adopters who value cutting-edge solutions and integration capabilities."
        ]
    },
    1: {
        "name": "Moderate Tech Usage",
        "profile": "Moderate tech users, including a mix of Retail and Services sectors.",
        "key_technologies": [
            "Analytics Tools: Usage of Facebook Pixel (24.24%) and Google Analytics (3.03%).",
            "JavaScript Libraries: Adoption of jQuery Migrate (10.61%) and jQuery UI (8.33%).",
            "Marketing Automation Tools: Some use of Klaviyo (3.03%)."
        ],
        "insights": [
            "These companies are at a mid-level in their technology adoption journey.",
            "They are likely looking for scalable solutions and support to advance their tech capabilities."
        ]
    },
    2: {
        "name": "Low Tech Usage",
        "profile": "Basic tech users, diverse industries with a significant number from Retail.",
        "key_technologies": [
            "CDN Technologies: Lower adoption rates, with Cloudflare at 10.08%.",
            "Analytics Tools: Limited usage of Facebook Pixel (9.80%) and Google Analytics (2.80%).",
            "JavaScript Libraries: Lower adoption of various libraries."
        ],
        "insights": [
            "These companies have minimal technology adoption and may need foundational solutions.",
            "They require simple, easy-to-implement technologies that provide immediate benefits."
        ]
    }
}

# Function to preprocess the input data
def preprocess_input(data):
    data.drop(columns=[col for col in columns_removed_during_training if col in data.columns], inplace=True)
    
    string_columns = data.select_dtypes(include=['object'])
    semicolon_columns = {column: string_columns[column].str.contains(';', regex=False).any() for column in string_columns.columns}
    semicolon_columns = {k: v for k, v in semicolon_columns.items() if v}

    for column in semicolon_columns:
        dummies = data[column].str.get_dummies(sep=';')
        dummies.columns = [f"{column}_{subcolumn.strip()}" for subcolumn in dummies.columns]
        data = pd.concat([data, dummies], axis=1)
        data.drop(column, axis=1, inplace=True)

    vertical_mapping = {
        'Manufacturing': ['Beverage Manufacturing', 'Manufacturing Toys, Edge, Tech', 'Pharmaceutical Manufacturing',
                          'Sporting Goods Manufacturing', 'Furniture and Home Furnishings Manufacturing', 'Medical Equipment Manufacturing',
                          'Apparel Manufacturing', 'Consumer Goods Manufacturing', 'Computers and Electronics Manufacturing', 'Tobacco Manufacturing'],
        'Retail': ['Salon - Hair & Beauty', 'E-Commerce DTC & Retail ', 'Retail', 'Retail, Apparel and Fashion',
                   'Retail Apparel and Fashion', 'Retail Luxury Goods and Jewelry', 'Apparel & Fashion', 'Retail Office Equipment',
                   'Retail Health and Personal Care Products', 'Retail ', 'Retail Motor Vehicles', 'Food and Beverage Retail',
                   'Wholesale Building Materials', 'Retail Furniture and Home Furnishings', 'Online and Mail Order Retail', 'Retail Art Dealers',
                   'Gifting business'],
        'Services': ['Strategic Communication Consultancy', 'Facilities Services', 'Environmental Services', 'IT Services and IT Consulting',
                     'Health, Wellness & Fitness', 'Wellness and Fitness Services', 'Utilities', 'Consumer Services',
                     'Business Consulting and Services', 'Design Services', 'Printing Services', 'Transportation, Logistics, Supply Chain and Storage'],
        'Healthcare': ['Medical Device', 'Hospitals and Health Care'],
        'Financial Services': ['Banking'],
        'Sports and Entertainment': ['Spectator Sports Leeds', 'Spectator Sports', 'Entertainment Providers', 'Sports Teams and Clubs'],
        'Technology': [''],
        'Others': ['Wholesale', 'Vehicle Repair and Maintenance', 'IKEA', 'Cosmetics', 'Restaurants', 'Building Materials', 'Artists and Writers']
    }
    
    reverse_vertical_map = {v: k for k, lst in vertical_mapping.items() for v in lst}
    if 'Current company vertical' in data.columns:
        data['Broad Vertical'] = data['Current company vertical'].map(reverse_vertical_map).fillna('Others')
    
    clustering_data = pd.get_dummies(data[['Broad Vertical']], drop_first=True)
    
    for col in expected_columns:
        if col not in clustering_data.columns:
            clustering_data[col] = 0
    
    clustering_data = clustering_data[expected_columns]
    
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    return clustering_data_scaled

# Function to predict clusters for new data
def predict_clusters(data):
    processed_data = preprocess_input(data)
    clusters = kmeans.predict(processed_data)
    return clusters

# Streamlit app
st.title('Clustering Model Deployment with Streamlit - Version 3.5')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.write("File uploaded successfully.")
    
    try:
        data = pd.read_csv(uploaded_file)
        st.write("**Input Data:**")
        st.write(data)

        # Make predictions
        clusters = predict_clusters(data)
        data['Cluster'] = clusters

        st.write("**Clustered Data:**")
        st.write(data)
        
        # Display which cluster each customer belongs to with insights
        for idx, cluster in enumerate(clusters):
            st.write(f"**Customer {idx + 1} is in Cluster {cluster}: {cluster_insights[cluster]['name']}**")
            st.write(f"**Profile:** {cluster_insights[cluster]['profile']}")
            st.write("**Key Technologies:**")
            for tech in cluster_insights[cluster]['key_technologies']:
                st.write(f" - {tech}")
            st.write("**Insights:**")
            for insight in cluster_insights[cluster]['insights']:
                st.write(f" - {insight}")
            st.write("---")

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.write("Upload a CSV file to see the results.")


# In[ ]:




