import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")
CSV_PATH = "../WA_Fn-UseC_-HR-Employee-Attrition.csv"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def migrate_csv_to_mongo():
    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    records = df.to_dict(orient="records")
    collection.delete_many({})
    result = collection.insert_many(records)
    print(f"✅ Inserted {len(result.inserted_ids)} documents into '{COLLECTION_NAME}' collection.")

if __name__ == "__main__":
    migrate_csv_to_mongo()
