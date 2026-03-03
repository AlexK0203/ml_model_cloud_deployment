import requests

URL = "https://ml-model-cloud-deployment-ff6d56f9a783.herokuapp.com/predict"

def test_live_api():
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
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
    response = requests.post(URL, json=data)
    # Debugging lines
    print(f"Status Code: {response.status_code}")
    print(f"Raw Response Text: {response.text}")

if __name__ == "__main__":
    test_live_api() 