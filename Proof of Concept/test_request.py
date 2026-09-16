import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "features": [0.1, 0.2, 0.3, 0.4, 0.5]
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
