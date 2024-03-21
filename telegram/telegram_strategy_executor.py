from lumibot.strategies import StrategyExecutor

class TelegramStrategyExecutor(StrategyExecutor):
    
    TELEGRAM_COMMAND = "telegram_command"

    def __init__(self, strategy, telegram_bot):
        super().__init__(strategy)
        self.telegram_bot = telegram_bot

    def get_queue(self):
        return self.queue

    def process_event(self, event, payload):
        # Process telegram events
        if event == self.TELEGRAM_COMMAND:
            command = payload["command"]
            if command == "/status":
                portfolio_value = self.strategy.get_portfolio_value()
                message = f"Current portfolio value is: {portfolio_value}"
                self.telegram_bot.send_message(message)
        
        # Process other events
        super().process_event(event, payload)