from lumibot.brokers import Alpaca
from telegram_strategy import TelegramStrategy
from lumibot.traders import Trader
from telegram_bot_handler import TelegramBotHandler
from credentials import AlpacaConfig

class ExampleStrategy(TelegramStrategy):
    def initialize(self):
        pass
    
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