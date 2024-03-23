from lumibot.brokers import Alpaca
from messaging_strategy import MessagingStrategy
from lumibot.traders import Trader
from telegram_bot_handler import TelegramBotHandler
from credentials import AlpacaConfig

class ExampleStrategy(MessagingStrategy):
    
    def before_starting_trading(self):
        self.send_message("before starting trading")

    def on_trading_iteration(self):
        self.send_message("on trading iteration")

        if self.first_iteration:
            self.order = self.create_order("SPY", 1, "buy")
            self.submit_order(self.order)

if __name__ == "__main__":
    live = True

    trader = Trader()
    broker = Alpaca(AlpacaConfig)
    strategy = ExampleStrategy(broker, symbol="SPY")

    # Set telegram bot and attach to strategy
    bot = TelegramBotHandler()
    strategy.set_messaging_bot(bot)

    # Set trader
    trader.add_strategy(strategy)
    trader.run_all()