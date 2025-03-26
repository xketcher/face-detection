import os
import cv2
import numpy as np
import logging
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters

# Load environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g., "https://your-app.onrender.com"

# Initialize FastAPI app
app = FastAPI()

# Initialize Telegram Bot Application
bot_app = Application.builder().token(BOT_TOKEN).build()
bot = Bot(token=BOT_TOKEN)

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

logging.basicConfig(level=logging.INFO)

async def detect_faces(image_path):
    """Detect faces in an image and return the processed image path."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    output_path = image_path.replace(".jpg", "_faces.jpg")
    cv2.imwrite(output_path, img)
    return output_path

async def handle_photo(update: Update, context):
    """Handle received images, detect faces, and send results."""
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"{file.file_id}.jpg"
    await file.download_to_drive(file_path)
    
    output_file = await detect_faces(file_path)

    chat_id = update.message.chat_id
    await bot.send_photo(chat_id=chat_id, photo=open(output_file, "rb"), caption="Face detection result")

# Add message handler
bot_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Webhook route for Telegram updates."""
    data = await request.json()
    update = Update.de_json(data, bot)
    await bot_app.process_update(update)
    return {"ok": True}

@app.get("/")
async def home():
    """Simple home route to check the server status."""
    return {"message": "Face Detection Bot is Running"}

async def set_webhook():
    """Set Telegram webhook if deployed on Render."""
    if WEBHOOK_URL:
        webhook_url = f"{WEBHOOK_URL}/webhook"
        await bot.set_webhook(webhook_url)
        logging.info(f"Webhook set: {webhook_url}")

if __name__ == "__main__":
    import asyncio

    # Set webhook if URL exists; otherwise, use polling
    if WEBHOOK_URL:
        asyncio.run(set_webhook())
    else:
        bot_app.run_polling()
