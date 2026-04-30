import requests
from bs4 import BeautifulSoup

session = requests.Session()
login_data = {'username': 'testuser', 'password': 'testpassword'}
session.post('http://127.0.0.1:5000/signup', data=login_data)
session.post('http://127.0.0.1:5000/login', data=login_data)

try:
    data = {'stock': 'ASIANPAINT.NS', 'range': '1Y'}
    res = session.post('http://127.0.0.1:5000/analyze', data=data)
    soup = BeautifulSoup(res.text, 'html.parser')
    found = False
    for h in soup.find_all('h6'):
        if "Tomorrow" in h.text and "Prediction" in h.text:
            found = True
            print("FOUND ML AI:")
            print(h.parent.text.strip())
    if not found:
        print("ML PREDICTION NOT FOUND on page. Server might have returned None or failed.")
        print(res.status_code)
except Exception as e:
    print(e)
