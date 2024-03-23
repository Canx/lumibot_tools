from aiogram import Bot, Dispatcher
from aiogram.types import Message
from credentials import TelegramConfig
import asyncio

class TelegramBotHandler:
    # TODO: Promocionar a clase abstracta, extraer la parte propia de Telegram.
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.telegram_token = TelegramConfig["TOKEN"]
        self.chat_id = TelegramConfig["CHAT_ID"]
        self.bot = Bot(token=self.telegram_token)
        self.dp = Dispatcher()
        # Registrar un manejador para interceptar todos los mensajes
        self.dp.message.register(self.handle_message)
        self.send_message_queue = asyncio.Queue()
        self.receive_message_queue = None

    # TODO: Promocionar a clase genérica
    def set_receive_message_queue(self, queue):
        self.receive_message_queue = queue

    # TODO: poner en clase genérica
    def send_message(self, text):
        coroutine = self.send_message_queue.put((self.chat_id, text))
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    # TODO: Promocionar a clase genérica??? Tendríamos que quitar el chat.id...
    async def process_send_message_queue(self):
        while True:
            chat_id, text = await self.send_message_queue.get()
            try:
                await self.bot.send_message(chat_id, text)
            except Exception as e:
                print(f"Error sending message to {chat_id}: {e}")
            self.send_message_queue.task_done()


    # TODO: Promocionar a clase genérica??? Tendríamos que quitar el chat.id...
    async def handle_message(self, message: Message):
        text = message.text
        if text.startswith('/'):  
            command = text.split()[0]
            chat_id = message.chat.id

            if self.receive_message_queue:
                self.receive_message_queue.put(("telegram_command", {"command": command, "chat_id": chat_id}))
         
    # Esto es propio de la estrategia. No tengo claro si el mensaje "Starting lumibot..." podríamos lanzarlo con self.send_message
    async def start(self):
        asyncio.create_task(self.process_send_message_queue())
        #await self.send_message_queue.put((self.chat_id, "Starting lumibot..."))
        self.send_message("Starting lumibot...")
        await self.dp.start_polling(self.bot, handle_signals=False)

    # TODO: Poner en clase abstracta, para sobreescribir. Llamado por messaging_strategy
    def start_bot_thread(self):
        self.loop.run_until_complete(self.start())
        self.loop.close()