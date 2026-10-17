import asyncio
import io
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile
from ultralytics import YOLO

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

MODEL_PATH = Path("weights/best.pt")
model = YOLO(str(MODEL_PATH))

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: Message):
    await message.answer(
        "Hello! Please send me an X-ray image, and I will detect potential fractures."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    status_msg = await message.answer("🔍 Analyzing the X-ray, please wait...")

    photo = message.photo[-1]
    file_io = io.BytesIO()
    await bot.download(photo, destination=file_io)
    file_io.seek(0)

    image = Image.open(file_io).convert("RGB")

    results = model.predict(source=image, conf=0.25, imgsz=1024, device=0)
    result = results[0]

    count = len(result.boxes)

    res_bgr = result.plot()
    res_rgb = Image.fromarray(res_bgr[..., ::-1])

    output_io = io.BytesIO()
    res_rgb.save(output_io, format="JPEG", quality=95)
    output_io.seek(0)

    if count == 0:
        caption = "✅ No fractures detected with high confidence."
    else:
        conf_list = [f"{float(box.conf[0]) * 100:.1f}%" for box in result.boxes]
        caption = (
            f"⚠️ Detected potential fracture zones: {count}\n"
            f"AI Confidence: {', '.join(conf_list)}"
        )

    input_file = BufferedInputFile(output_io.getvalue(), filename="result.jpg")
    await message.answer_photo(photo=input_file, caption=caption)
    await status_msg.delete()


@dp.message(F.text)
async def handle_text(message: Message):
    await message.answer(
        "I can only analyze images. Please send an X-ray photo to get started."
    )


async def main():
    print("Bot is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())