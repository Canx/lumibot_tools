from lumibot.strategies import StrategyExecutor

class TelegramStrategyExecutor(StrategyExecutor):
    
    TELEGRAM_COMMAND = "telegram_command"

    def __init__(self, strategy, telegram_bot):
        super().__init__(strategy)
        self.telegram_bot = telegram_bot

    def get_queue(self):
        return self.queue

    def process_event(self, event, payload):
        # Primero, procesa eventos estándar de la estrategia
        super().process_event(event, payload)
        
        # Ahora, maneja los eventos específicos de Telegram
        if event == self.TELEGRAM_COMMAND:
            # Aquí, procesas el comando recibido desde Telegram.
            # Por ejemplo, podrías querer chequear el tipo de comando
            # y actuar en consecuencia.
            command = payload["command"]
            chat_id = payload["chat_id"]
            if command == "/status":
                # Obtén el estado de la estrategia
                portfolio_value = self.strategy.get_portfolio_value()
                message = f"El valor actual del portafolio es: {portfolio_value}"
                # Usa la instancia del bot de Telegram para enviar el mensaje
                self.telegram_bot.send_message(message)