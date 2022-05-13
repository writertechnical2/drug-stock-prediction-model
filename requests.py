import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Name of drug':5, 'period in months':200, 'sales_in_second_month':400})

print(r.json())