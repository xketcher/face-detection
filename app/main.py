import os
import cv2
import subprocess
from telegram.ext import Updater, MessageHandler, Filters

# Get bot token from environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image_path):
    """Detect faces in an image and return the processed image path."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    output_path = image_path.replace(".jpg", "_faces.jpg")
    cv2.imwrite(output_path, img)
    return output_path

def handle_photo(update, context):
    """Handle received images, detect faces, and send results using curl."""
    photo = update.message.photo[-1].get_file()
    file_path = f"{photo.file_id}.jpg"
    photo.download(file_path)
    
    output_file = detect_faces(file_path)

    chat_id = update.message.chat_id
    subprocess.run([
        "curl", "-s", "-X", "POST",
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        "-F", f"chat_id={chat_id}",
        "-F", f"photo=@{output_file}",
        "-F", "caption=Face detection result"
    ])

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
