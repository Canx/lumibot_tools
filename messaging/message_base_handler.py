from abc import ABC, abstractmethod
import asyncio
import threading

class MessageBaseHandler(ABC):
    def __init__(self):
        self.send_message_queue = asyncio.Queue()
        self.receive_message_queue = None
        self.thread = None
        self.loop = None

    def send_message(self, text):
        coroutine = self.send_message_queue.put(text)
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def process_send_message_queue(self):
        while True:
            text = await self.send_message_queue.get()
            try:
                await self.send_message_to_platform(text)
            except Exception as e:
                print(f"Error sending message: {e}")
            self.send_message_queue.task_done()

    def set_receive_message_queue(self, queue):
        self.receive_message_queue = queue
    
    def start_handler_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start())
        self.loop.close()

    @abstractmethod
    async def send_message_to_platform(self, text):
        pass
    
    @abstractmethod
    async def handle_incoming_message(self, message):
        pass

    @abstractmethod
    async def start(self):
        pass