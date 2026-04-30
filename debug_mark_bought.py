from web_app import app, db, User, BuyList

with app.app_context():
    user = User.query.first()
    if not user:
        user = User(username="test", password="123")
        db.session.add(user)
        db.session.commit()
        
    b = BuyList.query.filter_by(user_id=user.id, status="Queued").first()
    if not b:
        b = BuyList(user_id=user.id, stock="RELIANCE.NS", status="Queued")
        db.session.add(b)
        db.session.commit()

with app.test_client() as client:
    with client.session_transaction() as sess:
        sess["user"] = user.username
        sess["user_id"] = user.id
    
    print("Before:", BuyList.query.filter_by(user_id=user.id, stock=b.stock).first().status)
    res = client.post('/mark_bought', data={
        "stock": b.stock,
        "quantity": "25",
        "purchase_price": "2500.50"
    })
    print("STATUS", res.status_code)
    
    after = BuyList.query.filter_by(user_id=user.id, stock=b.stock).first()
    print("After:", after.status, after.quantity, after.purchase_price)
