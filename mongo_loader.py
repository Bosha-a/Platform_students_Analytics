import streamlit as st
import pandas as pd
from pymongo import MongoClient


# Connection

@st.cache_resource
def get_client() -> MongoClient:
    """Return a cached MongoClient.  Reads MONGO_URI from st.secrets."""
    uri = st.secrets["MONGO_URI"]
    return MongoClient(uri)


def get_db(db_name: str = "kayfa_analytics"):
    """Return the target database object."""
    return get_client()[db_name]


# Collection loader

@st.cache_data(ttl=3600, show_spinner=False)
def load_collection(name: str, db_name: str = "kayfa_analytics") -> pd.DataFrame:
    db = get_db(db_name)
    docs = list(db[name].find({}, {"_id": 0}))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    return df
