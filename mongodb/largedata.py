import pandas as pd
from pymongo import MongoClient
import gc # Garbage Collector

def ingest_high_volume(file_path):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["nyc_taxi_analytics"]
    collection = db["November_2025"]

    print("Loading Parquet file into memory...")
    df = pd.read_parquet(file_path)
    
    # Lower the batch size to 5,000 to keep memory usage stable
    batch_size = 5000 
    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        # Use a slice to create a small chunk
        batch = df.iloc[i : i + batch_size].to_dict('records')
        
        try:
            collection.insert_many(batch)
            print(f"Uploaded records {i} to {i + len(batch)}...")
        except Exception as e:
            print(f"Error at record {i}: {e}")
            break
        
        # EXPLICITLY clear the batch from memory and call garbage collector
        del batch
        gc.collect() 

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_high_volume("data.parquet")