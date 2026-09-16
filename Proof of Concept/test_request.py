import requests

url = "http://127.0.0.1:5000/predict"

# Replace this with a real feature vector matching what your
# scaler/PCA/model were trained on (same length and order).
payload = {
    "features": [0.1, 0.2, 0.3, 0.4, 0.5]
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())