import json

FILE = "users.json"

def load_users():
    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(FILE, "w") as f:
        json.dump(users, f)

def signup(username, password):

    # validation
    if username.strip() == "" or password.strip() == "":
        return False, "Username and Password cannot be empty"

    if len(password) < 4:
        return False, "Password must be at least 4 characters"

    users = load_users()

    if username in users:
        return False, "User already exists"

    users[username] = password
    save_users(users)
    return True, "Account created successfully"


def login(username, password):
    users = load_users()

    if username in users and users[username] == password:
        return True
    return False
