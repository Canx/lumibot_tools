from lumibot.strategies import Strategy
from telegram_strategy_executor import TelegramStrategyExecutor
import threading

class TelegramStrategy(Strategy):
        
    def set_telegram_bot(self, telegram_bot):
        self.telegram_bot = telegram_bot
        self._executor = TelegramStrategyExecutor(self, self.telegram_bot)
        self.broker._add_subscriber(self._executor)
        self.telegram_bot.set_receive_message_queue(self._executor.get_queue())

        # Init telegram bot in its own thread
        self.bot_thread = threading.Thread(target=self.telegram_bot.start_bot_thread, daemon=False)
        self.bot_thread.start()

    def send_message(self, text):
        if self.telegram_bot:
            self.telegram_bot.send_message(text)
        else:
            self.logger.error("Telegram bot not configured.")

    def status_command(self, parameters=None):
        portfolio_value = self.get_portfolio_value()
        message = f"Current portfolio value is: {portfolio_value}"
        
        return message
    
    def hola_command(self, parameters=None):
        return "Hola"