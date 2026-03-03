from fastapi.testclient import TestClient
from main import app

client = TestClient(app) 

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Census Income Prediction API"}

def test_post_predict_below():
    # Define a data sample likely to result in <=50K
    data = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 34145,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

def test_post_predict_above ():
    # Define a data sample likely to result in >50K
    data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 34145,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}