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

if __name__ == "__main__":
    live = True

    trader = Trader()
    broker = Alpaca(AlpacaConfig)
    strategy = ExampleStrategy(broker, symbol="SPY", account_history_db_connection_str="sqlite:///stats.db")

    # Set telegram bot and attach to strategy
    bot = TelegramBotHandler(TelegramConfig)
    strategy.set_messaging_bot(bot)

    # Set trader
    trader.add_strategy(strategy)
    trader.run_all()
