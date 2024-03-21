from lumibot.strategies import Strategy
from telegram_strategy_executor import TelegramStrategyExecutor
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from telegram_bot_handler import TelegramBotHandler
from credentials import AlpacaConfig
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
            
    def before_starting_trading(self):
        self.send_message("before starting trading")

    def on_trading_iteration(self):
        self.send_message("on trading iteration")

if __name__ == "__main__":
    live = True

    trader = Trader()
    broker = Alpaca(AlpacaConfig)
    strategy = TelegramStrategy(broker, symbol="SPY")

    # Set telegram bot and attach to strategy
    bot = TelegramBotHandler()
    strategy.set_telegram_bot(bot)

    # Set trader
    trader.add_strategy(strategy)
    trader.run_all()
