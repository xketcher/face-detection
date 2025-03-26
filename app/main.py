import os
import cv2
import dlib
import numpy as np
import logging
import asyncio
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Load environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Example: "https://your-bot.onrender.com"

# Initialize FastAPI app
app = FastAPI()

# Initialize Telegram Bot Application
bot_app = Application.builder().token(BOT_TOKEN).build()
bot = Bot(token=BOT_TOKEN)

# Load OpenCV and dlib face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

logging.basicConfig(level=logging.INFO)

# Store received images
received_images = []


def extract_face_landmarks(image_path):
    """Extract face landmarks from an image."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None, None

    shape = predictor(gray, faces[0])
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    return landmarks, img


def swap_faces(image1_path, image2_path):
    """Swap faces between two images."""
    landmarks1, img1 = extract_face_landmarks(image1_path)
    landmarks2, img2 = extract_face_landmarks(image2_path)

    if landmarks1 is None or landmarks2 is None:
        return None  # No face detected

    # Compute the convex hull (mask) around the face
    hull1 = cv2.convexHull(landmarks1)
    hull2 = cv2.convexHull(landmarks2)

    # Create face masks
    mask1 = np.zeros_like(img1, dtype=np.uint8)
    mask2 = np.zeros_like(img2, dtype=np.uint8)
    cv2.fillConvexPoly(mask1, hull1, (255, 255, 255))
    cv2.fillConvexPoly(mask2, hull2, (255, 255, 255))

    # Extract face region from image2 and resize to fit image1
    face2 = cv2.bitwise_and(img2, mask2)
    bbox = cv2.boundingRect(hull2)
    x, y, w, h = bbox
    face2_resized = cv2.resize(face2[y:y+h, x:x+w], (w, h))

    # Place the resized face from image2 onto image1
    img1[y:y+h, x:x+w] = face2_resized

    output_path = "swapped_faces.jpg"
    cv2.imwrite(output_path, img1)
    return output_path


async def handle_photo(update: Update, context):
    """Handle received images and swap faces when two are received."""
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"{file.file_id}.jpg"
    await file.download_to_drive(file_path)

    received_images.append(file_path)

    if len(received_images) == 2:
        output_file = swap_faces(received_images[0], received_images[1])

        if output_file:
            await bot.send_photo(chat_id=update.message.chat_id, photo=open(output_file, "rb"),
                                 caption="🔄 Face swap result!")
        else:
            await update.message.reply_text("⚠️ Face detection failed. Try sending clearer photos.")

        received_images.clear()  # Reset for next pair


async def start_command(update: Update, context):
    """Handle /start command."""
    await update.message.reply_text("👋 Hello! Send me 2 photos, and I'll swap faces between them!")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Webhook route for Telegram updates."""
    data = await request.json()
    update = Update.de_json(data, bot)

    await bot_app.initialize()
    await bot_app.process_update(update)

    return {"ok": True}


@app.get("/")
async def home():
    """Simple home route to check the server status."""
    return {"message": "Face Detection & Swap Bot is Running"}


async def set_webhook():
    """Set Telegram webhook if deployed on Render."""
    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL}/webhook"
        await bot.set_webhook(webhook_url)
        logging.info(f"Webhook set: {webhook_url}")


if __name__ == "__main__":
    if WEBHOOK_URL:
        asyncio.run(set_webhook())
    else:
        bot_app.run_polling()
