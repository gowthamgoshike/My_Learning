from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import status
from fastapi import HTTPException
from fastapi import Depends

app = FastAPI()
#GET Request
@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

#Query Parameters
@app.get("/search")
def search(q: str, limit: int = 10):
    return {"query": q, "limit": limit}

#POST Request

class User(BaseModel):
    name: str
    age: int
    city: str

@app.post("/users")
def create_user(user: User):
    return {"message": "User created", "user": user}

#Path + Query + Body
@app.put("/users/{user_id}")
def update_user(user_id: int, user: User, active: bool = True):
    return {
        "user_id": user_id,
        "active": active,
        "user": user
    }

#Status Codes

@app.post("/login", status_code=status.HTTP_201_CREATED)
def login():
    return {"message": "Logged in"}

#Error Handling

@app.get("/divide")
def divide(a: int, b: int):
    if b == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero")
    return {"result": a / b}

#Dependency Injection
def get_db():
    return "DB Connected"

@app.get("/data")
def get_data(db=Depends(get_db)):
    return {"db": db}
