import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('cricket_dataset_test.csv', 'rb')}

response = requests.post(url, files=files)
print(response.json())