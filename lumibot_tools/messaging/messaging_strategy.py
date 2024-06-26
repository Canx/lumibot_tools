from lumibot.strategies import Strategy
from lumibot.entities import Asset
from lumibot_tools.messaging.messaging_strategy_executor import MessagingStrategyExecutor
from decimal import Decimal

class MessagingStrategy(Strategy):

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messaging_bot = None

    # Sets the messaging bot and its executor, then subscribes the executor to the broker.
    # Also starts the messaging bot in its own thread.    
    def set_messaging_bot(self, messaging_bot):
        self.messaging_bot = messaging_bot
        self._executor = MessagingStrategyExecutor(self, self.messaging_bot)
        self.broker._add_subscriber(self._executor)
        self.messaging_bot.set_receive_message_queue(self._executor.get_queue())
        self.messaging_bot.start_handler_thread()

    # Sends a message using the messaging bot. Logs an error if the bot is not configured.
    def send_message(self, text):
        if self.is_backtesting:
            return
        if isinstance(text, str):
            if self.messaging_bot is not None:
                self.messaging_bot.send_message(text)
        
    def log_message(self, message, color=None, broadcast=False):
        super().log_message(message, color, broadcast)  # Llama al método original para asegurar que el log se registre
        
        # Verifica si el modo de backtesting está desactivado
        if not self.is_backtesting:
            # Envia el mensaje al bot de mensajería
            self.send_message(message)        

    # Returns the current portfolio value as a message.
    def status_command(self, parameters=None):
        portfolio_value = self.get_portfolio_value()
        message = f"Current portfolio value is: {portfolio_value}"
        
        return message
    
    def help_command(self, parameters=None):
        # Filtrar todos los métodos que terminan en '_command'
        commands = [method for method in dir(self) if method.endswith('_command') and callable(getattr(self, method))]
        # Remover el sufijo '_command' y formatear para el mensaje de ayuda
        command_list = [command[:-8] for command in commands]
        # Crear un mensaje de ayuda
        help_message = "Available commands: " + ", ".join(command_list)
        return help_message
    
    # Retrieves and formats the current positions, including their value and percent of the portfolio.
    def positions_command(self, parameters=None):
        positions = self.get_positions()

        positions_details_list = []
        for position in positions:
            if position.asset == self.quote_asset:
                last_price = 1
            else:
                last_price = self.get_last_price(position.asset)

            if last_price is None or not isinstance(last_price, (int, float, Decimal)):
                self.logger.info(f"Last price for {position.asset} is not a number: {last_price}")
                continue

            position_value = position.quantity * last_price

            if position.asset.asset_type == "option":
                position_value = position_value * 100

            percent_of_portfolio = position_value / self.portfolio_value

            positions_details_list.append(
                {
                    "asset": position.asset,
                    "quantity": position.quantity,
                    "value": position_value,
                    "percent_of_portfolio": percent_of_portfolio,
                }
            )

        positions_details_list = sorted(positions_details_list, key=lambda x: x["percent_of_portfolio"], reverse=True)

        positions_text = ""
        for position in positions_details_list:
            # positions_text += f"{position.quantity:,.2f} {position.asset} ({percent_of_portfolio:,.0%})\n"
            positions_text += (
                f"{position['quantity']:,.2f} {position['asset']} ({position['percent_of_portfolio']:,.0%})\n"
            )

        # TODO: Not working... {returns_text} removed from message
        # returns_text, stats_df = self.calculate_returns()

        message = f"""
                **Update for {self.name}**
                **Account Value:** ${self.portfolio_value:,.2f}
                **Cash:** ${self.get_cash():,.2f}
                **Positions:**
                {positions_text}
                """

        message = "\n".join(line.lstrip() for line in message.split("\n"))

        return message
            
    
    # Logs and sends a message when an order is filled, detailing the trade and its impact on the portfolio.
    def on_filled_order(self, position, order, price, quantity, multiplier):
        
        portfolio_value = self.get_portfolio_value()

        order_value = price * float(quantity)

        # If option, multiply % of portfolio by 100
        if order.asset.asset_type == Asset.AssetType.OPTION:
            order_value = order_value * 100

        # Calculate the percent of the portfolio that this position represents
        percent_of_portfolio = order_value / portfolio_value

        # Capitalize the side
        side = order.side.capitalize()

        # Check if we are buying or selling
        if side == "Buy":
            emoji = "🟢📈 "
        else:
            emoji = "🔴📉 "

        # Create a message to send
        message = f"""
                {emoji} {side} {quantity:,.2f} {position.asset} @ ${price:,.2f} ({percent_of_portfolio:,.0%} of the account)
                Trade Total = ${order_value:,.2f}
                Account Value = ${portfolio_value:,.0f}
                """        
        
        self.log_message(message)
