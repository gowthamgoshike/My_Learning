# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 1. Define the Data Model
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

# 2. Simulated Database (Dictionary)
fake_db = {}

@app.post("/items/{item_id}", status_code=201)
def create_item(item_id: int, item: Item):
    if item_id in fake_db:
        raise HTTPException(status_code=400, detail="Item already exists")
    fake_db[item_id] = item
    return {"item_id": item_id, "name": item.name, "saved": True}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]