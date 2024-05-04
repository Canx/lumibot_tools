import discord
from lumibot_tools.messaging.message_base_handler import MessageBaseHandler
import asyncio

class DiscordBotHandler(MessageBaseHandler):
    def __init__(self, discord_config):
        super().__init__()
        self.discord_token = discord_config["TOKEN"]
        self.channel_id = discord_config["CHANNEL"]
        self.channel = None
        self.command_prefix = '/'
        # Crear el cliente de Discord
        intents = discord.Intents.default()  # Incluye la mayoría de los intents no privilegiados
        intents.guilds = True
        intents.messages = True  # Para recibir mensajes
        intents.message_content = True  # Para recibir el contenido de los mensajes
        self.client = discord.Client(intents=intents)

        # Añadir el manejador de eventos de mensajes
        self.client.event(self.on_message)

    async def on_message(self, message):
        # Evitar procesar mensajes del propio bot
        if message.author == self.client.user:
            return
        # Procesa el mensaje
        await self.handle_incoming_message(message)

    async def handle_incoming_message(self, message):
        # Implementa tu lógica específica aquí, por ejemplo:
        text = message.content
        if text.startswith(self.command_prefix):
            command = text.split()[0].lstrip(self.command_prefix)
            parameters = ' '.join(text.split()[1:])
            if self.receive_message_queue:
                self.receive_message_queue.put(("message_command", {
                    "command": command,
                    "parameters": parameters,
                    "chat_id": message.author.id
                }))

    async def init_channel(self):
        if self.channel is None:
            self.channel = self.client.get_channel(self.channel_id)
            if self.channel is None:
                self.channel = await self.client.fetch_channel(self.channel_id)
        
    async def send_message_to_platform(self, text):
        # Asume que channel_id siempre se proporciona
        await self.init_channel()
        if self.channel:
            await self.channel.send(text)

    async def start(self):
        # Inicia el cliente de Discord
        asyncio.create_task(self.process_send_message_queue())
        await self.client.start(self.discord_token)



import discord
# from discord.ext import commands
# from lumibot_tools.messaging.message_base_handler import MessageBaseHandler
# import asyncio

# class DiscordBotHandler(MessageBaseHandler):
#     def __init__(self, discord_config):
#         super().__init__()
#         self.discord_token = discord_config["TOKEN"]
#         self.channel_id = discord_config["CHANNEL"]
#         self.command_prefix = '/'

#         # Definir los intents necesarios
#         intents = discord.Intents.default()  # Incluye la mayoría de los intents no privilegiados
#         intents.messages = True  # Para recibir mensajes
#         intents.message_content = True  # Para recibir el contenido de los mensajes

#         # Inicializar el bot con los intents
#         self.bot = commands.Bot(command_prefix='/', intents=intents)

#         self.bot.add_listener(self.handle_incoming_message)

#     #async def setup_hook(self):
#     #    # Crea tareas aquí
#     #    self.bot.loop.create_task(self.process_send_message_queue())

#     async def send_message_to_platform(self, text):
#         # Asume que channel_id siempre se proporciona
#         channel = self.bot.get_channel(self.channel_id)
#         if channel:
#             await channel.send(text)
    
#     async def handle_incoming_message(self, message):
#         # Implementación del método abstracto
#         text = message.content
#         if text.startswith(self.command_prefix):
#             command = text.split()[0]
#             parameters = ' '.join(text.split()[1:])
#             if self.receive_message_queue:
#                 self.receive_message_queue.put(("message_command", {
#                     "command": command.lstrip('!'),
#                     "parameters": parameters,
#                     "chat_id": message.author.id
#                 }))


#     async def start(self):
#         # Asegúrate de llamar a super().start() si está definido en la clase base
#         asyncio.create_task(self.process_send_message_queue())
#         await self.bot.start(self.discord_token)
