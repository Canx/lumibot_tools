from lumibot.strategies import Strategy
from lumibot.entities import Asset
from messaging_strategy_executor import MessagingStrategyExecutor
import threading
from decimal import Decimal

class MessagingStrategy(Strategy):
        
    def set_messaging_bot(self, messaging_bot):
        self.messaging_bot = messaging_bot
        self._executor = MessagingStrategyExecutor(self, self.messaging_bot)
        self.broker._add_subscriber(self._executor)
        self.messaging_bot.set_receive_message_queue(self._executor.get_queue())

        # Init messaging bot in its own thread
        self.bot_thread = threading.Thread(target=self.messaging_bot.start_bot_thread, daemon=False)
        self.bot_thread.start()

    def send_message(self, text):
        if self.messaging_bot:
            self.messaging_bot.send_message(text)
        else:
            self.logger.error("Messaging bot not configured.")

    def status_command(self, parameters=None):
        portfolio_value = self.get_portfolio_value()
        message = f"Current portfolio value is: {portfolio_value}"
        
        return message
    
    def hola_command(self, parameters=None):
        return "Hola"
    
    def positions_command(self, parameters=None):
        # Get the current positions
        positions = self.get_positions()

        # Create the positions text
        positions_details_list = []
        for position in positions:
            # Check if the position asset is the quote asset

            if position.asset == self.quote_asset:
                last_price = 1
            else:
                # Get the last price
                last_price = self.get_last_price(position.asset)

            # Make sure last_price is a number
            if last_price is None or not isinstance(last_price, (int, float, Decimal)):
                self.logger.info(f"Last price for {position.asset} is not a number: {last_price}")
                continue

            # Calculate teh value of the position
            position_value = position.quantity * last_price

            # If option, multiply % of portfolio by 100
            if position.asset.asset_type == "option":
                position_value = position_value * 100

            # Calculate the percent of the portfolio that this position represents
            percent_of_portfolio = position_value / self.portfolio_value

            # Add the position details to the list
            positions_details_list.append(
                {
                    "asset": position.asset,
                    "quantity": position.quantity,
                    "value": position_value,
                    "percent_of_portfolio": percent_of_portfolio,
                }
            )

        # Sort the positions by the percent of the portfolio
        positions_details_list = sorted(positions_details_list, key=lambda x: x["percent_of_portfolio"], reverse=True)

        # Create the positions text
        positions_text = ""
        for position in positions_details_list:
            # positions_text += f"{position.quantity:,.2f} {position.asset} ({percent_of_portfolio:,.0%})\n"
            positions_text += (
                f"{position['quantity']:,.2f} {position['asset']} ({position['percent_of_portfolio']:,.0%})\n"
            )

        # TODO: Not working... {returns_text} removed from message
        # returns_text, stats_df = self.calculate_returns()

        # Create a message to send to Messaging bot (round the values to 2 decimal places)
        message = f"""
                **Update for {self.name}**
                **Account Value:** ${self.portfolio_value:,.2f}
                **Cash:** ${self.get_cash():,.2f}
                **Positions:**
                {positions_text}
                """

        # Remove any leading whitespace
        # Remove the extra spaces at the beginning of each line
        message = "\n".join(line.lstrip() for line in message.split("\n"))

        return message
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        # Get the portfolio value
        portfolio_value = self.get_portfolio_value()

        # Calculate the value of the position
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
        
        self.send_message(message)