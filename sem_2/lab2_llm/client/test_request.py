import requests
import json

url = "http://localhost:8000/approve"

client_data = {
    "age": 35,
    "income": 55000,
    "education": "bachelor",
    "married": "yes"
}

response = requests.post(url, json=client_data)
print("Status:", response.status_code)
print("Response:", response.json())