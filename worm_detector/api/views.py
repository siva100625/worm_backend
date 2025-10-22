from django.shortcuts import render
import os
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import traceback
import bcrypt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from keras.models import load_model
from keras.layers import RandomFlip
# =============================
# Configuration
# =============================
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["earthworm", "flatworm"]
client=MongoClient("mongodb+srv://testUser:TSX8uNGoZc2AuteR@cluster0.7zpc1wb.mongodb.net/") 
db = client["worm_detector_db"] 
predictions_collection = db["predictions"]
# =============================
# Load Model (.h5 safe loading)
# =============================
MODEL_PATH = os.path.join(settings.BASE_DIR, "api", "ml_models", "best_worm_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

from tensorflow import MobileNetV2
from tensorflow import models, layers

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Load weights safely
model.load_weights(MODEL_PATH)
print("✅ Model successfully rebuilt and loaded!")


# =============================
# Prediction Endpoint
# =============================
from django.core.mail import send_mail
def save_uploaded_file_to_temp(uploaded_file):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return temp_path


def save_bytes_to_temp(data_bytes):
    temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(temp_fd, 'wb') as f:
        f.write(data_bytes)
    return temp_path


def preprocess_image_file(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

@csrf_exempt
def predict_image(request):
    temp_path = None
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST allowed"}, status=405)

        # Get username from request (optional, from frontend)
        username = request.POST.get("username", None)

        # Receive image
        if request.FILES.get("image"):
            temp_path = save_uploaded_file_to_temp(request.FILES["image"])
        elif request.content_type and request.content_type.startswith("image/"):
            temp_path = save_bytes_to_temp(request.body)
        else:
            return JsonResponse({"error": "No image provided"}, status=400)

        # Preprocess
        img_batch = preprocess_image_file(temp_path)

        # Predict (Binary)
        pred = model.predict(img_batch)[0][0]
        if pred > 0.5:
            predicted_class = "flatworm"
            confidence = float(pred)
        else:
            predicted_class = "earthworm"
            confidence = float(1 - pred)

        # Timestamp (IST)
        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        ist_timestamp = ist_time.strftime("%Y-%m-%d %H:%M:%S")

        # Save to MongoDB
        record = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "timestamp": ist_timestamp,
            "username": username
        }
        predictions_collection.insert_one(record)

        # === Send Email if Flatworm Detected ===
        db = client["user_database"]
        users_collection = db["users"]
        if predicted_class == "flatworm":
            print("jo")
            try:
                user = users_collection.find_one({"username": username})
                if user:
                    print(f"🔹 Found user in DB: {user}")
                else:
                    print(f"❌ No user found with username: {username}")

                if user and "email" in user and user["email"]:
                    user_email = user["email"]
                    print(f"🔹 Sending email to: {user_email}")

                    subject = "⚠️ Flatworm Detected!"
                    message = (
                        f"Hello {username},\n\n"
                        f"Our system detected a *flatworm* in your submitted image.\n\n"
                        f"Confidence: {confidence:.2f}\n"
                        f"Time: {ist_timestamp}\n\n"
                        f"Please review this immediately.\n\n"
                        f"Regards,\nWorm Detection System 🪱"
                    )

                    sent_count = send_mail(
                        subject,
                        message,
                        settings.DEFAULT_FROM_EMAIL,
                        [user_email],
                        fail_silently=False
                    )

                    if sent_count == 1:
                        print(f"✅ Email successfully sent to {user_email}")
                    else:
                        print(f"❌ send_mail returned {sent_count}")
                else:
                    print(f"❌ User has no email or email field missing")
            except Exception as mail_err:
                print(f"🔥 Email sending exception: {mail_err}")

        return JsonResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3),
            "message": "Prediction successful!"
        })

    except Exception as e:
        print("🔥 Prediction Error Traceback:")
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# =============================
# Authentication Endpoints
# =============================
db = client["user_database"]
users_collection = db["users"]
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    username = request.data.get("username")
    password = request.data.get("password")
    email = request.data.get("email")

    if not username or not password or not email:
        return Response({"error": "Username, password, and email required"}, status=status.HTTP_400_BAD_REQUEST)

    if users_collection.find_one({"username": username}):
        return Response({"error": "Username already exists"}, status=status.HTTP_400_BAD_REQUEST)

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({
        "username": username,
        "password": hashed_pw.decode("utf-8"),
        "email": email
    })

    return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    username = request.data.get("username")
    password = request.data.get("password")

    if not username or not password:
        return Response({"error": "Username and password required"}, status=status.HTTP_400_BAD_REQUEST)

    user = users_collection.find_one({"username": username})
    if not user:
        return Response({"error": "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)

    stored_hashed_pw = user["password"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), stored_hashed_pw):
        return Response({"error": "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)

    return Response({"message": "Logged in successfully"}, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([AllowAny])
def logout(request):
    """Simple logout endpoint — clears client-side session."""
    return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)

# =============================
# All Predictions Endpoint
# =============================
@csrf_exempt
def all_predictions(request):
    try:
        data = list(predictions_collection.find({}, {"_id": 0}).sort("_id", -1))
        if not data:
            return JsonResponse({"message": "No predictions yet."}, status=404)
        return JsonResponse({"predictions": data}, safe=False)
    except Exception as e:
        print("🔥 Fetch Predictions Error:")
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)
