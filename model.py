import os

from data import load_data, preproc_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from google.cloud import storage 
import pickle
from dotenv import load_dotenv

def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = make_pipeline(LogisticRegression(random_state=42, max_iter=500))
    model.fit(X_train, y_train)
    print('model trained 💪')
    predictions = model.predict(X_test)
    print(f"Accuracy of the trained model on test: {model.score(X_test, y_test)}")
    return model

def save_model(trained_model):
    load_dotenv()
    bucket_name = os.getenv("GCP_BUCKET")
    object_name = "model.pkl"
    local_file = "model.pkl"

    with open(local_file, 'wb') as f:
        pickle.dump(trained_model, f)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    if blob.exists() == False:
        with open(local_file, "wb") as f:
            pickle.dump(trained_model, f)
        print('Model saved localy 💾') 

        blob.upload_from_filename(local_file)
        print(f"File uploaded to gs://{bucket_name}/{object_name} ✅")   
    else:
        blob.download_to_filename(local_file)
        with open(local_file, "rb") as f:
            preprocessor = pickle.load(f)
        print(f'Model loaded localy from GS: {bucket_name}/{object_name}') 


def load_model(GCS_model_name):
    bucket_name = "titanic_model_2025_02_07"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(GCS_model_name)

    if blob.exists() == False:
        print(f'no model "{GCS_model_name}" saved ❌')
    else:
        blob.download_to_filename(GCS_model_name)
        with open(GCS_model_name, "rb") as f:
            model = pickle.load(f)
        print(f'Model saved locally and loaded ✅')
        return model
    

if __name__ == "__main__":
    df = load_data(table_name = 'RAW_train_data')
    X_processed, y = preproc_data(df)
    trained_model = train_model(X_processed,y)
    save_model(trained_model)
    model = load_model('model.pkl')





