import requests
import csv

base_url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"

api_key = "ee9d4cf30f1141c9a5f145748240803"

# Parameters 
params = {
    "q": "Mumbai, India",
    "date": "2023-12-17",
    "enddate": "2024-01-01",  
    "tp": "1",  # 1-hour intervals
    "format": "csv",  
    "key": api_key  
}

# Sending GET request
response = requests.get(base_url, params=params)

# Checking if the request was successful (status code 200)
if response.status_code == 200:
    with open('weathe1r.csv', 'wb') as csvfile:
        csvfile.write(response.content)
else:
    print("Failed to retrieve data. Status code:", response.status_code)