import requests

# ----------------------------
# CONFIG
# ----------------------------
BASE_URL = "http://127.0.0.1:8000/api"  # API prefix
USERNAME = "vivek sir"
PASSWORD = "mypassword123"
IMAGE_PATH = "31.jpeg"  # put your test image path here

# ----------------------------
# 1️⃣ SIGNUP
# ----------------------------
signup_url = f"{BASE_URL}/signup/"
signup_data = {"username": USERNAME, "password": PASSWORD}

print("=== SIGNUP ===")
resp = requests.post(signup_url, json=signup_data)
print(resp.status_code, resp.text)

# ----------------------------
# 2️⃣ LOGIN
# ----------------------------
login_url = f"{BASE_URL}/login/"
login_data = {"username": USERNAME, "password": PASSWORD}

print("\n=== LOGIN ===")
resp = requests.post(login_url, json=login_data)
print(resp.status_code, resp.text)

# ----------------------------
# 3️⃣ PREDICTION
# ----------------------------
predict_url = f"{BASE_URL}/predict/"

print("\n=== PREDICTION ===")
with open(IMAGE_PATH, "rb") as img_file:
    files = {"image": img_file}
    resp = requests.post(predict_url, files=files)
    print(resp.status_code, resp.text)

# ----------------------------
# 4️⃣ ALL PREDICTIONS
# ----------------------------
all_pred_url = f"{BASE_URL}/all/"

print("\n=== ALL PREDICTIONS ===")
resp = requests.get(all_pred_url)
print(resp.status_code, resp.text)
