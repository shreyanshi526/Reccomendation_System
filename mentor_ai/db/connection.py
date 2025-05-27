from pymongo import MongoClient
from mentor_ai.utils.config import MONGO_URI, DATABASE_NAME

def get_db():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    return db
