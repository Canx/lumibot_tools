from abc import ABC, abstractmethod
import asyncio
import threading
import time

class MessageBaseHandler(ABC):

    # Initializes queues for sending and receiving messages, and sets up threading.
    def __init__(self):
        self.send_message_queue = asyncio.Queue()
        self.receive_message_queue = None
        self.thread = None
        self.loop = None

    # Enqueues a message to be sent asynchronously.
    def send_message(self, text):
        coroutine = self.send_message_queue.put(text)
        while self.loop is None:
            time.sleep(0.1)  # Pequeña pausa para evitar saturación
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    # Processes the queue of messages to be sent, calling a platform-specific method to actually send each message.
    async def process_send_message_queue(self):
        while True:
            text = await self.send_message_queue.get()
            try:
                await self.send_message_to_platform(text)
            except Exception as e:
                print(f"Error sending message: {e}")
            self.send_message_queue.task_done()

    # Sets the queue for receiving incoming messages from the strategy executor.
    def set_receive_message_queue(self, queue):
        self.receive_message_queue = queue
    
    # Starts the message handler in its own thread.
    def start_handler_thread(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    # Sets up and runs the asyncio event loop for this thread.
    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start())
        self.loop.close()

    # Abstract method to be implemented by subclasses for sending messages using specific platform APIs.
    @abstractmethod
    async def send_message_to_platform(self, text):
        pass
    
    # Abstract method for handling incoming messages, to be implemented by subclasses.
    @abstractmethod
    async def handle_incoming_message(self, message):
        pass

    # Abstract method for starting message processing, to be implemented by subclasses.
    @abstractmethod
    async def start(self):
        pass
