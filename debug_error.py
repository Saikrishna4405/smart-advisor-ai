from web_app import app
import builtins

with app.test_client() as client:
    with client.session_transaction() as sess:
        sess["user"] = "test"
        sess["user_id"] = 1
    
    response = client.get('/buy')
    print("Status:", response.status_code)
    if response.status_code != 200:
        print(response.get_data(as_text=True))
