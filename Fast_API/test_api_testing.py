
# test_main.py
from fastapi.testclient import TestClient
from api_testing import app  # Import your FastAPI app

client = TestClient(app)

# Test 1: Happy Path - Successfully creating an item
def test_create_item():
    payload = {"name": "Laptop", "price": 999.99, "is_offer": True}
    
    # Send a POST request
    response = client.post("/items/1", json=payload)
    
    # Assertions
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Laptop"
    assert data["saved"] is True
    assert data["item_id"] == 1

# Test 2: Happy Path - Reading the item we just created
# Updated test_read_item
def test_read_item():
    # Use a unique ID (e.g., 99) to avoid clashing with Item 1 from the previous test
    client.post("/items/99", json={"name": "Mouse", "price": 40.00})
    
    response = client.get("/items/99")
    assert response.status_code == 200
    # Now we expect what we just created for ID 99 (is_offer defaults to None)
    assert response.json() == {"name": "Mouse", "price": 40.00, "is_offer": None}
# Test 3: Edge Case - Try to create an item that already exists
def test_create_duplicate_item():
    # Setup: Create item 2
    client.post("/items/2", json={"name": "Phone", "price": 500})
    
    # Action: Try to create item 2 AGAIN
    response = client.post("/items/2", json={"name": "Phone", "price": 500})
    
    # Assertions: Expecting a 400 Bad Request
    assert response.status_code == 400
    assert response.json() == {"detail": "Item already exists"}

# Test 4: Edge Case - Get an item that doesn't exist
def test_read_nonexistent_item():
    response = client.get("/items/999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}

# Test 5: Validation Error - Send bad data (missing 'price')
def test_create_item_invalid_data():
    # Sending payload without 'price'
    payload = {"name": "Tablet"} 
    
    response = client.post("/items/3", json=payload)
    
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422