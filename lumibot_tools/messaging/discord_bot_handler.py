import discord
from discord.ext import commands
from lumibot_tools.messaging.message_base_handler import MessageBaseHandler
import asyncio

class DiscordBotHandler(MessageBaseHandler):
    def __init__(self, discord_config):
        super().__init__()
        self.discord_token = discord_config["TOKEN"]
        self.guild_id = discord_config["GUILD_ID"]  # Opcional, dependiendo de si necesitas interactuar con un servidor específico
        self.bot = commands.Bot(command_prefix='/')

        @self.bot.event
        async def on_ready():
            print(f'Logged in as {self.bot.user.name}')

        @self.bot.command()
        async def handle_incoming_command(ctx, *args):
            command = ctx.invoked_with
            parameters = ' '.join(args)
            if self.receive_message_queue:
                self.receive_message_queue.put(("message_command", {
                    "command": command,
                    "parameters": parameters,
                    "chat_id": ctx.author.id
                }))

    async def send_message_to_platform(self, text, channel_id):
        channel = self.bot.get_channel(channel_id)
        if channel:
            await channel.send(text)
        else:
            print("Channel not found")

    async def start(self):
        asyncio.create_task(self.process_send_message_queue())
        await self.bot.start(self.discord_token)
