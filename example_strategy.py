from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
from config import ALPACA_CONFIG, TELEGRAM_CONFIG

class ExampleStrategy(MessagingStrategy):
    
    def initialize(self):
        self.assets = []
    def before_starting_trading(self):
        self.send_message("before starting trading")

    def hola_command(self, parameters=None):
        return "hola"
    
    def add_command(self, parameters=None):
        self.assets.append(parameters)
        return ', '.join(self.assets)

    def remove_command(self, parameters=None):
        self.assets.remove(parameters)
        return ', '.join(self.assets)

    def on_trading_iteration(self):
        self.log_message("on trading iteration")

        if self.first_iteration:
            self.order = self.create_order("SPY", 1, "buy")
            self.submit_order(self.order)
            self.log_message("buyed 1 SPY")

if __name__ == "__main__":
    live = True

    trader = Trader()
    broker = Alpaca(ALPACA_CONFIG)
    strategy = ExampleStrategy(broker)

    # Set telegram bot and attach to strategy
    bot = TelegramBotHandler(TELEGRAM_CONFIG)
    strategy.set_messaging_bot(bot)

    # Set trader
    trader.add_strategy(strategy)
    trader.run_all()
