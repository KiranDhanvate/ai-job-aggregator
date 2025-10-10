import pandas as pd
import numpy as np
import re
import pickle
import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Constants ---
MAX_VOCAB_SIZE = 10000
MAX_JOB_DESC_LEN = 250
MAX_USER_SKILLS_LEN = 50
ARTIFACTS_PATH = './recommendation_model/artifacts/'
MOCK_DATA_PATH = './'
JOBS_DF_PATH = ARTIFACTS_PATH + 'jobs_df.pkl' # We'll save the processed jobs df

def clean_text(text):
    """A simple function to clean text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_scraped_json(json_data):
    """Converts the JSON from the scraper into a clean pandas DataFrame."""
    # If json_data is a string, parse it
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    jobs_df = pd.DataFrame(data['jobs'])
    print(f"Loaded {len(jobs_df)} jobs from JSON.")
    
    # Use the unique job 'id' from the scraper as the job_id
    jobs_df.rename(columns={'id': 'job_id'}, inplace=True)
    
    # Ensure essential columns exist
    if 'description' not in jobs_df.columns:
        raise ValueError("JSON must contain a 'description' field for each job.")
        
    return jobs_df

def create_mock_data_and_interactions(jobs_df):
    """Creates mock users and interactions based on the provided jobs DataFrame."""
    print("Creating mock user and interaction data...")
    users = {
        'user_id': [1, 2, 3, 4, 5],
        'skills': [
            'python machine learning tensorflow fastapi django pandas',
            'javascript react node.js html css typescript aws',
            'java spring boot microservices sql docker kubernetes',
            'data analysis sql tableau power bi python statistics',
            'devops aws terraform ansible jenkins ci/cd'
        ],
        'location_preference': ['Bengaluru, Karnataka, India', 'Pune, Maharashtra, India', 'Hyderabad, Telangana, India', 'Remote', 'Bengaluru East, Karnataka, India']
    }
    users_df = pd.DataFrame(users)
    users_df.to_csv(MOCK_DATA_PATH + 'users.csv', index=False)
    
    interactions = []
    for _, user in users_df.iterrows():
        user_skills = set(user['skills'].split())
        for _, job in jobs_df.sample(frac=0.2).iterrows(): # Sample 20% of jobs
            job_desc = clean_text(job['description'])
            job_skills = set(job_desc.split())
            
            if len(user_skills.intersection(job_skills)) > 2:
                interactions.append({'user_id': user['user_id'], 'job_id': job['job_id'], 'clicked': 1})
            elif np.random.rand() < 0.05:
                 interactions.append({'user_id': user['user_id'], 'job_id': job['job_id'], 'clicked': 0})

    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(MOCK_DATA_PATH + 'interactions.csv', index=False)
    print(f"Created {len(users_df)} mock users and {len(interactions_df)} interactions.")
    return users_df, interactions_df


def load_and_prepare_data(scraped_data_json):
    """Main function to load, merge, and preprocess all data for TRAINING."""
    print("Loading and preparing data for training...")
    
    # Process the input JSON to get a jobs DataFrame
    jobs_df = process_scraped_json(scraped_data_json)
    
    # Create mock data based on these jobs
    users_df, interactions_df = create_mock_data_and_interactions(jobs_df)
        
    # Merge data
    df = pd.merge(interactions_df, users_df, on='user_id')
    df = pd.merge(df, jobs_df, on='job_id')

    # Clean text fields
    df['description'] = df['description'].apply(clean_text)
    df['skills'] = df['skills'].apply(clean_text)

    # --- Feature Engineering & Encoding ---
    all_text = pd.concat([df['description'], df['skills']]).unique()
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<UNK>')
    tokenizer.fit_on_texts(all_text)

    location_encoder = LabelEncoder()
    # Fit on all possible locations from jobs AND users to handle unseen values at prediction time
    all_locations = pd.concat([df['location'], df['location_preference']]).unique().astype(str)
    location_encoder.fit(all_locations)
    
    df['location_encoded'] = location_encoder.transform(df['location'])
    
    # Save artifacts
    if not os.path.exists(ARTIFACTS_PATH):
        os.makedirs(ARTIFACTS_PATH)
        
    with open(ARTIFACTS_PATH + 'tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
    with open(ARTIFACTS_PATH + 'location_encoder.pkl', 'wb') as f: pickle.dump(location_encoder, f)
    # Save the processed jobs dataframe for use in prediction
    jobs_df.to_pickle(JOBS_DF_PATH)

    print("Artifacts saved (tokenizer, encoder, jobs_df).")

    # Prepare inputs for the model
    X_job_desc = pad_sequences(tokenizer.texts_to_sequences(df['description']), maxlen=MAX_JOB_DESC_LEN)
    X_user_skills = pad_sequences(tokenizer.texts_to_sequences(df['skills']), maxlen=MAX_USER_SKILLS_LEN)
    
    X = {
        'job_description_input': X_job_desc,
        'user_skills_input': X_user_skills,
        'location_input': df['location_encoded'].values
    }
    y = df['clicked'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), y, test_size=0.2, random_state=42)
    
    X_train_dict = {key: np.array(X_train[key].tolist()) for key in X_train.columns}
    X_test_dict = {key: np.array(X_test[key].tolist()) for key in X_test.columns}

    return X_train_dict, X_test_dict, y_train, y_test, tokenizer, location_encoder