# lumibot_tools

- messaging: attach a messaging bot to a strategy
- signals: true/false signals based on technical indicators
- bin/run.py: script to run strategies easily

## Install & Upgrade

```
pip install --upgrade git+https://github.com/Canx/lumibot_tools.git
```

## Messaging

### Features

- Redirects log_message to bot
- Implement easily bot commands in Strategy adding a method *_command(parameters)

### Usage

Example strategy that inherits from MessagingStrategy class and attach TelegramBotHandler to the strategy
```
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
from config import ALPACA_CONFIG, TELEGRAM_CONFIG

class ExampleStrategy(MessagingStrategy):

    def initialize(self):
        self.assets = []

    def hola_command(self, parameters=None):
        return "hola"

    def add_command(self, parameters=None):
        self.assets.append(parameters)
        return ', '.join(self.assets)

    def remove_command(self, parameters=None):
        self.assets.remove(parameters)
        return ', '.join(self.assets)

    def on_trading_iteration(self):
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
```

### Suported platforms

- Telegram
- Discord (TODO)

Add to your config.py file:

```
TELEGRAM_CONFIG = {
    "TOKEN": "<TELEGRAM TOKEN>",
    "CHAT_ID": "<TELEGRAM CHAT ID>"
}
```

## Signals

- check implemented signals in code [lumibot_tools/signals/trading_signals.py](./lumibot_tools/signals/trading_signals.py)
### Usage

```

from lumibot_tools.signals import Signals

class ExampleStrategy(Strategy):

    def initialize(self):
        self.signals = Signals(self)

    def entry_signal(self, symbol):
        return (
            self.signals.ma_cross_with_atr_validation(symbol, short_length=12, long_length=25, ma_type='EMA', cross='bullish', atr_length=14, atr_factor=1) and
            self.signals.short_over_long_ma(symbol, short_length=50, long_length=200, ma_type='EMA')
        )
    
    def exit_signal(self, position):
        return (
            self.signals.ma_crosses(position.symbol, short_length=21, long_length=55, ma_type='EMA', cross='bearish')
        )

    def on_trading_iteration(self):
        self.check_exits(self.exit_signal)
        self.check_entries(self.entry_signal)

    def check_entries(self, signal_function):
        for symbol in self.assets:
                if signal_function(symbol):
                    self.enter_position(symbol)
 
    def check_exits(self, signal_function):
        for position in self.get_positions():
            if signal_function(position):
                    self.exit_position(position)
    
```
