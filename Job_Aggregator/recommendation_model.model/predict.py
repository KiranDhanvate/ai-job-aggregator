import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from recommendation_model.model import FactorizationMachine # Important
from recommendation_model.preprocess import clean_text, process_scraped_json

# --- Load Artifacts ONCE when the module is loaded ---
ARTIFACTS_PATH = './recommendation_model/artifacts/'
try:
    MODEL = load_model(
        ARTIFACTS_PATH + 'convfm_model.h5',
        custom_objects={'FactorizationMachine': FactorizationMachine}
    )
    with open(ARTIFACTS_PATH + 'tokenizer.pkl', 'rb') as f:
        TOKENIZER = pickle.load(f)
    with open(ARTIFACTS_PATH + 'location_encoder.pkl', 'rb') as f:
        LOCATION_ENCODER = pickle.load(f)
    
    # Also load the mock users to select from
    USERS_DF = pd.read_csv('./users.csv')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    print("Please run `python -m recommendation_model.train` first.")
    MODEL = None


def get_recommendations_for_scraped_jobs(user_id: int, scraped_jobs_json: dict, top_k: int = 10):
    """
    Generates top_k job recommendations for a user from LIVE scraped data.
    """
    if MODEL is None:
        return {"error": "Model not loaded. Please train the model first."}
        
    try:
        user_profile = USERS_DF[USERS_DF['user_id'] == user_id].iloc[0]
    except IndexError:
        return {"error": f"User with ID {user_id} not found."}

    # --- Process the live scraped data ---
    jobs_df = process_scraped_json(scraped_jobs_json)
    jobs_df['description_clean'] = jobs_df['description'].apply(clean_text)
    
    # --- Preprocess User and Job Data for Prediction ---
    user_skills_seq = pad_sequences(TOKENIZER.texts_to_sequences([user_profile['skills']]), maxlen=50)

    # Handle locations that the encoder hasn't seen before
    # We assign them to an "unknown" category (index 0)
    job_locations_encoded = []
    for loc in jobs_df['location'].astype(str):
        try:
            job_locations_encoded.append(LOCATION_ENCODER.transform([loc])[0])
        except ValueError:
            job_locations_encoded.append(0) # Assign to a default/unknown index
            
    job_locations_encoded = np.array(job_locations_encoded)
    job_desc_seq = pad_sequences(TOKENIZER.texts_to_sequences(jobs_df['description_clean']), maxlen=250)
    
    num_jobs = len(jobs_df)
    user_skills_repeated = np.repeat(user_skills_seq, num_jobs, axis=0)

    # --- Predict ---
    predictions = MODEL.predict({
        'job_description_input': job_desc_seq,
        'user_skills_input': user_skills_repeated,
        'location_input': job_locations_encoded
    })
    
    jobs_df['score'] = predictions
    
    # --- Rank and Return ---
    recommended_jobs = jobs_df.sort_values(by='score', ascending=False).head(top_k)
    
    return recommended_jobs[['job_id', 'title', 'company', 'location', 'score']].to_dict(orient='records')