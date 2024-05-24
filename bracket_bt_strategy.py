
from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset
from lumibot.strategies import Strategy
from lumibot.backtesting import PolygonDataBacktesting
import datetime
import pandas as pd
from math import floor

"""
"""
class TestStrategy(Strategy):

    def initialize(self, parameters=None):
        self.sleeptime = "1D"

    def on_trading_iteration(self):


        if self.first_iteration:
            current_price = self.get_last_price("SPY")
            self.log_message(f"SPY price: {current_price}")
            order = self.create_order(
                asset=Asset("SPY"),
                quantity=10,
                side="buy",
                limit_price=current_price,
                stop_loss_price=current_price*0.50,
                type="limit")
            
            self.submit_order(order)
                
    def _print_order(self, order):
        self.log_message(f"Order id: {order.identifier}")
        self.log_message(f"Order status: {order.status}" )
        self.log_message(f"Order time_in_force: {order.time_in_force}")
        self.log_message(f"Order class: {order.order_class}")
        self.log_message(f"Order type: {order.type}")
        self.log_message(f"Order direction: {order.side}")
        self.log_message(f"Order stop-loss: {order.stop_loss_price}") 
        self.log_message(f"Custom params: {order.custom_params}")

    def print_order(self, order):
        self._print_order(order)

        if order.dependent_order:
            self.log_message(f"dependent order:")
            self._print_order(order.dependent_order)  

    def on_canceled_order(self, order):
        self.log_message(f"{order} has been canceled by the broker")

    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.log_message(position)
        self.log_message(f"Quantity: {quantity}" )
        self.log_message(f"Price: {price}")
        
        self.print_order(order)
        
if __name__ == "__main__":
        from config import POLYGON_CONFIG
        backtesting_start = datetime.datetime(2023, 6, 1)
        backtesting_end = datetime.datetime(2023, 12, 1)
        trader = Trader(backtest=True)
        trading_fee = TradingFee(percent_fee=0.001)
        TestStrategy.backtest(
            PolygonDataBacktesting,
            backtesting_start,
            backtesting_end,
            polygon_api_key=POLYGON_CONFIG["KEY"],
            polygon_has_paid_subscription=POLYGON_CONFIG["PAID"],
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            benchmark_asset="SPY",
            risk_free_rate=0.532)
