import os
import cv2
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

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

logging.basicConfig(level=logging.INFO)

# Store received images
received_images = []


def detect_face(image):
    """Detect a face in the image and return the face region + bounding box."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]  # Take the first detected face
    face_region = image[y:y+h, x:x+w]
    
    return face_region, (x, y, w, h)


def swap_faces(image1_path, image2_path):
    """Swap faces between two images using OpenCV."""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    face1, bbox1 = detect_face(img1)
    face2, bbox2 = detect_face(img2)

    if face1 is None or face2 is None:
        return None  # No face detected

    # Resize faces to match each other's size
    face1_resized = cv2.resize(face1, (bbox2[2], bbox2[3]))
    face2_resized = cv2.resize(face2, (bbox1[2], bbox1[3]))

    # Swap faces
    img1[bbox1[1]:bbox1[1]+bbox1[3], bbox1[0]:bbox1[0]+bbox1[2]] = face2_resized
    img2[bbox2[1]:bbox2[1]+bbox2[3], bbox2[0]:bbox2[0]+bbox2[2]] = face1_resized

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
    return {"message": "Face Swap Bot is Running"}


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
