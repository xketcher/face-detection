import os
import cv2
import logging
import asyncio
from fastapi import FastAPI, Request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters

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


async def start_command(update: Update, context):
    """Handle /start command."""
    await update.message.reply_text("👋 Hello! I am a Face Detection Bot.\nSend me a photo, and I'll detect faces!")


async def help_command(update: Update, context):
    """Handle /help command."""
    await update.message.reply_text("📖 *Commands:*\n"
                                    "/start - Start the bot\n"
                                    "/help - Show this help message\n"
                                    "/about - About this bot\n"
                                    "📷 Send a photo, and I'll detect faces!")


async def about_command(update: Update, context):
    """Handle /about command."""
    await update.message.reply_text("🤖 *Face Detection Bot*\n"
                                    "Built with OpenCV and FastAPI.\n"
                                    "Deploys on Render.\n"
                                    "👨‍💻 Developer: You!")


async def echo_text(update: Update, context):
    """Echo back any text messages sent by users."""
    await update.message.reply_text(f"You said: {update.message.text}")


# Add handlers to the bot
bot_app.add_handler(CommandHandler("start", start_command))
bot_app.add_handler(CommandHandler("help", help_command))
bot_app.add_handler(CommandHandler("about", about_command))
bot_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_text))  # Echo all text


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Webhook route for Telegram updates."""
    data = await request.json()
    update = Update.de_json(data, bot)

    # Ensure application is initialized before processing the update
    await bot_app.initialize()
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
    # Start webhook or polling based on environment
    if WEBHOOK_URL:
        asyncio.run(set_webhook())
    else:
        bot_app.run_polling()
