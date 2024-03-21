from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from credentials import TelegramConfig
import asyncio

class TelegramBotHandler:
    def __init__(self):
        asyncio.get_event_loop().set_debug(True)
        self.loop = None
        self.telegram_token = TelegramConfig["TOKEN"]
        self.chat_id = TelegramConfig["CHAT_ID"]
        self.bot = Bot(token=self.telegram_token)
        self.dp = Dispatcher()
        self.dp.message.register(self.status_command, Command(commands=["status"]))
        self.send_message_queue = asyncio.Queue()

    def set_receive_message_queue(self, queue):
        self.receive_message_queue = queue

    def send_message(self, text):
        coroutine = self.send_message_queue.put((self.chat_id, text))
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def process_send_message_queue(self):
        while True:
            chat_id, text = await self.send_message_queue.get()
            try:
                await self.bot.send_message(chat_id, text)
            except Exception as e:
                print(f"Error sending message to {chat_id}: {e}")
            self.send_message_queue.task_done()

    async def status_command(self, message: Message, state: FSMContext):
        chat_id = message.chat.id
        self.receive_message_queue.put(("telegram_command", {"command": "/status", "chat_id": chat_id}))

    async def start(self):
        asyncio.create_task(self.process_send_message_queue())
        await self.send_message_queue.put((self.chat_id, "Starting lumibot..."))
        await self.dp.start_polling(self.bot, handle_signals=False)

    # This is called from TelegramStrategy in a new thread
    def start_bot_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        loop.run_until_complete(self.start())
        loop.close()