from web_app import app, db, User, BuyList

with app.app_context():
    user = User.query.first()
    if not user:
        user = User(username="test", password="123")
        db.session.add(user)
        db.session.commit()
        
    b = BuyList.query.filter_by(user_id=user.id).first()
    if not b:
        b = BuyList(user_id=user.id, stock="RELIANCE.NS", quantity=10, purchase_price=3000.0, status="Bought")
        db.session.add(b)
        db.session.commit()

with app.test_client() as client:
    with client.session_transaction() as sess:
        sess["user"] = user.username
        sess["user_id"] = user.id
    res = client.get('/buy')
    print("STATUS", res.status_code)
    if res.status_code != 200:
        print(res.get_data(as_text=True))
