import requests

# create predictions from ride information
ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

# send post requests to endpoint to get prediction
url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())

