from aiogram import Bot, Dispatcher
from aiogram.types import Message
from lumibot_tools.messaging.message_base_handler import MessageBaseHandler
import asyncio

class TelegramBotHandler(MessageBaseHandler):

    def __init__(self, telegram_config):
        super().__init__()
        self.telegram_token = telegram_config["TOKEN"]
        self.chat_id = telegram_config["CHAT_ID"]
        self.bot = Bot(token=self.telegram_token)
        self.dp = Dispatcher()
        self.dp.message.register(self.handle_incoming_message)

    async def send_message_to_platform(self, text):
        await self.bot.send_message(chat_id=self.chat_id, text=text)            

    async def handle_incoming_message(self, message: Message):
        text = message.text
        if text.startswith('/'):
            parts = text.split()
            command = parts[0]
            parameters = ' '.join(parts[1:])  # Junta los parámetros restantes en una sola cadena
            if self.receive_message_queue:
                self.receive_message_queue.put(("message_command", {"command": command, "parameters": parameters, "chat_id": message.chat.id}))

    async def start(self):
        asyncio.create_task(self.process_send_message_queue())
        await self.dp.start_polling(self.bot, handle_signals=False)
