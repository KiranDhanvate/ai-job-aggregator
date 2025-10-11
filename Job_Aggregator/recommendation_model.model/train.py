import tensorflow as tf
import json
from recommendation_model.preprocess import load_and_prepare_data
from recommendation_model.model import build_convfm_model

# --- Model Configuration ---
MODEL_CONFIG = {
    'embedding_dim': 32,
    'latent_dim': 16,
    'filters': 64,
    'kernel_size': 3,
    'fm_dim': 10,
    'max_job_len': 250,
    'max_user_len': 50
}
ARTIFACTS_PATH = './recommendation_model/artifacts/'
MODEL_SAVE_PATH = ARTIFACTS_PATH + 'convfm_model.h5'
# Define the path to your scraped data
SCRAPED_DATA_JSON_PATH = 'scraped_jobs.json' # <-- IMPORTANT: Put your JSON output here

def main():
    """Main training pipeline."""
    # 1. Load the scraped JSON data
    print(f"Loading scraped data from {SCRAPED_DATA_JSON_PATH}...")
    try:
        with open(SCRAPED_DATA_JSON_PATH, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {SCRAPED_DATA_JSON_PATH} was not found.")
        print("Please run a scrape and save the output to that file.")
        return

    # 2. Load and preprocess data
    X_train, X_test, y_train, y_test, tokenizer, location_encoder = load_and_prepare_data(scraped_data)

    vocab_size = len(tokenizer.word_index) + 1
    num_locations = len(location_encoder.classes_)

    # 3. Build the model
    print("Building the ConvFM model...")
    model = build_convfm_model(
        vocab_size=vocab_size,
        num_locations=num_locations,
        config=MODEL_CONFIG
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # 4. Train the model
    print("\n--- Starting Model Training ---")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        batch_size=64, epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping], verbose=1
    )

    # 5. Evaluate and save
    print("\n--- Evaluating & Saving Model ---")
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]:.4f}, Test AUC: {results[2]:.4f}")
    model.save(MODEL_SAVE_PATH)
    
    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()