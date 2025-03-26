import os
import shutil
from fastapi import FastAPI, UploadFile, File
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.utils.executor import start_polling
import asyncio
from spleeter.separator import Separator
from pydub import AudioSegment

TOKEN = os.getenv("BOT_TOKEN")  # Render Environment Variable
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# FastAPI app setup
app = FastAPI()

# Spleeter separator (2 stems: Vocals + Instrumental)
separator = Separator("spleeter:2stems-lite")

async def process_audio(input_path, output_path):
    """Process the audio file to remove vocals and keep only the instrumental."""
    separator.separate_to_file(input_path, "output")

    # Extract instrumental file
    instrumental_path = f"output/{os.path.splitext(os.path.basename(input_path))[0]}/accompaniment.wav"
    
    # Convert to MP3
    sound = AudioSegment.from_wav(instrumental_path)
    sound.export(output_path, format="mp3")

    return output_path

@dp.message_handler(content_types=[ContentType.AUDIO, ContentType.VOICE])
async def handle_audio(message: types.Message):
    """Handle incoming audio files and process them."""
    file_id = message.audio.file_id if message.audio else message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path

    # Download the file
    input_audio = f"input/{file_id}.ogg"
    output_audio = f"output/{file_id}.mp3"
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    await bot.download_file(file_path, input_audio)

    # Process the file to remove vocals
    await process_audio(input_audio, output_audio)

    # Send back the instrumental version
    with open(output_audio, "rb") as audio_file:
        await message.reply_audio(audio_file)

    # Cleanup files
    os.remove(input_audio)
    os.remove(output_audio)

@app.on_event("startup")
async def startup_event():
    """Start the Telegram bot on FastAPI startup."""
    loop = asyncio.get_event_loop()
    loop.create_task(dp.start_polling())

@app.get("/")
def read_root():
    return {"message": "Telegram Music Bot Running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload an audio file and process it via API."""
    input_audio = f"input/{file.filename}"
    output_audio = f"output/{file.filename.replace('.ogg', '.mp3')}"
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    with open(input_audio, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the file to remove vocals
    await process_audio(input_audio, output_audio)

    return {"message": "Processing complete", "output_file": output_audio}
