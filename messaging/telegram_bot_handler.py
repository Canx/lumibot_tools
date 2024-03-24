from aiogram import Bot, Dispatcher
from aiogram.types import Message
from credentials import TelegramConfig
import asyncio
import threading

class TelegramBotHandler:
    # TODO: Promocionar a clase abstracta, extraer la parte propia de Telegram.
    def __init__(self):
        self.thread = None
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
                self.receive_message_queue.put(("message_command", {"command": command, "chat_id": chat_id}))

    def start_handler_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start())
        self.loop.close()

    # Esto es propio de la estrategia. No tengo claro si el mensaje "Starting lumibot..." podríamos lanzarlo con self.send_message
    async def start(self):
        asyncio.create_task(self.process_send_message_queue())
        #await self.send_message_queue.put((self.chat_id, "Starting lumibot..."))
        self.send_message("Starting lumibot...")
        await self.dp.start_polling(self.bot, handle_signals=False)


    
