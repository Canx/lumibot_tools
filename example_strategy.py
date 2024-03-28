from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from messaging.messaging_strategy import MessagingStrategy
from messaging.telegram_bot_handler import TelegramBotHandler
from credentials import AlpacaConfig, TelegramConfig

class ExampleStrategy(MessagingStrategy):
    
    def before_starting_trading(self):
        self.send_message("before starting trading")

    def hola_command(self, parameters=None):
        return "hola"
    
    def on_trading_iteration(self):
        self.send_message("on trading iteration")

        if self.first_iteration:
            self.order = self.create_order("SPY", 1, "buy")
            self.submit_order(self.order)
            self.send_message("buyed 1 SPY")

if __name__ == "__main__":
    live = True

    trader = Trader()
    broker = Alpaca(AlpacaConfig)
    strategy = ExampleStrategy(broker)

    # Set telegram bot and attach to strategy
    bot = TelegramBotHandler(TelegramConfig)
    strategy.set_messaging_bot(bot)

    # Set trader
    trader.add_strategy(strategy)
    trader.run_all()
