from lumibot.strategies import StrategyExecutor

class MessagingStrategyExecutor(StrategyExecutor):
    
    MESSAGE_COMMAND = "message_command"

    def __init__(self, strategy, messaging_bot):
        super().__init__(strategy)
        self.messaging_bot = messaging_bot

    def get_queue(self):
        return self.queue

    def process_event(self, event, payload):
        # Process telegram events
        if event == self.MESSAGE_COMMAND:
            command = payload["command"].lstrip('/')
            method_name = f"{command}_command"

            if hasattr(self.strategy, method_name):
                method = getattr(self.strategy, method_name)
                message = method(payload.get("parameters", {}))
                
                if message:
                    self.messaging_bot.send_message(message)
            else:
                self.messaging_bot.send_message(f"Command {command} not recognized.")
        
        # Process other events
        else:
            super().process_event(event, payload)