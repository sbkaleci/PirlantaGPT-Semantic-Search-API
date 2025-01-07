import requests

url = "http://127.0.0.1:5000/query"
headers = {"Content-Type": "application/json"}
data = {"query": "Gulen egitim konusunda ne dusunuyor?"}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)