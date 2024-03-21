from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from credentials import TelegramConfig
import asyncio
import time

class TelegramBotHandler:
    def __init__(self, loop=None):
        asyncio.get_event_loop().set_debug(True)
        self.loop = loop or asyncio.get_event_loop()
        self.telegram_token = TelegramConfig["TOKEN"]
        self.chat_id = TelegramConfig["CHAT_ID"]

        self.bot = Bot(token=self.telegram_token)
        self.dp = Dispatcher()

        # Registra el manejador de mensajes
        self.dp.message.register(self.status_command, Command(commands=["status"]))

        # Cola de envio de mensajes
        self.send_message_queue = asyncio.Queue()

    # Cola de repeción de mensajes (StrategyExecutor)
    def set_receive_message_queue(self, queue):
        self.receive_message_queue = queue

    # Método para encolar mensajes, se llamará desde la estrategia
    def send_message(self, text):
        start_time = time.time()
        coroutine = self.send_message_queue.put((self.chat_id, text))
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        #result = future.result()  # Esto es bloqueante
        end_time = time.time()
        print(f"Tiempo transcurrido: {end_time - start_time} segundos")

    async def process_send_message_queue(self):
        while True:
            print("Antes de llamar a send_message_queue")
            chat_id, text = await self.send_message_queue.get()
            print("Después de llamar a send_message_queue")
            try:
                await self.bot.send_message(chat_id, text)
            except Exception as e:
                print(f"Error al enviar mensaje a {chat_id}: {e}")
            self.send_message_queue.task_done()

    async def status_command(self, message: Message, state: FSMContext):
        chat_id = message.chat.id
        print(f"Comando /status recibido de {chat_id}")
        # Aquí puedes hacer lo que necesites con el comando /status
        self.receive_message_queue.put(("telegram_command", {"command": "/status", "chat_id": chat_id}))

    async def start(self):
        # Tarea para procesar los mensajes para enviar
        asyncio.create_task(self.process_send_message_queue())
        await self.send_message_queue.put((self.chat_id, "Arrancando..."))
        # Inicia el polling como una tarea de asyncio
        await self.dp.start_polling(self.bot, handle_signals=False)