from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset
from lumibot.backtesting import PolygonDataBacktesting
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
from config import ALPACA_CONFIG, TELEGRAM_CONFIG, POLYGON_CONFIG
import datetime

"""
Estrategia que monitorizará los activos que tenemos y
nos dirá si tenemos que vender o comprar por Telegram
"""
class MonitorStrategy(MessagingStrategy):

    def initialize(self):
        self.sleeptime = "1D"
        self.assets = self.get_position_symbols()

    def on_trading_iteration(self):
        if self.first_iteration:
            # Repartimos el capital para comprar los diferentes símbolos.
            self.comprar_equitativamente()   

        else:
            # 1. Verificamos las posibles salidas de cada símbolo.
            self.check_exits()

            # 2. Verificamos las posibles entradas de cada símbolo.
            self.check_entries()

    # TODO: Quiero que se busquen los assets en los que no tenemos posición.
    # Para estos comprobamos la señal de compra, que será entrar si tenemos un máximo de x días
    # y nos posicionamos de la siguiente forma:
    # Repartimos el cash disponible entre el número de assets en los que no tenemos posición.

    def check_entries(self, days=50):
        all_symbols = self.get_all_symbols()
        
        # Obtenemos la lista de todas las posiciones y filtramos aquellas con cantidad igual a 0 o que no existen.
        current_positions = {position.symbol: position for position in self.get_positions()}
        unowned_or_zero_assets = [symbol for symbol in all_symbols if symbol not in current_positions or current_positions[symbol].quantity == 0]

        for symbol in unowned_or_zero_assets:
            self.buy_like_turtletraders(symbol, days)
    
    def buy_like_turtletraders(self, symbol, days=50):
        # Obtener los precios históricos para el símbolo dado
        historical_prices = self.get_historical_prices(symbol, length=days)
        prices = historical_prices.df if historical_prices else None

        if prices is not None and 'close' in prices.columns:
            # Comprobar si el último precio es un máximo en los últimos 'days' días
            latest_price = prices['close'].iloc[-1]
            max_days = prices['close'].tail(days).max()
            if latest_price == max_days:
                self.log_message(f"{symbol} has reached a new {days}-day high at {latest_price}, preparing to buy.")
                # Calcular la cantidad de acciones a comprar basado en el cash disponible
                investment_per_asset = self.cash / self.calculate_number_of_eligible_assets()
                shares_to_buy = int(investment_per_asset / latest_price)

                if shares_to_buy > 0:
                    order = self.create_order(symbol, shares_to_buy, "buy")
                    self.submit_order(order)
                    self.log_message(f"Placed order for {shares_to_buy} shares of {symbol} at price {latest_price}.")
                else:
                    self.log_message(f"Not enough cash to buy {symbol}.")
            else:
                self.log_message(f"No buy signal for {symbol} as price {latest_price} is not the {days}-day high.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")

    def calculate_number_of_eligible_assets(self):
        # Podría ajustarse para recalcular dinámicamente el número de activos elegibles en diferentes escenarios
        return len(self.get_all_symbols())  # As an example, adjust based on actual eligibility logic
    

    def check_exits(self):
        for position in self.get_positions():
            asset = position.asset

            self.log_message("Asset:" + asset.symbol)
            if asset.asset_type != Asset.AssetType.STOCK:
                self.log_message("No es stock!")
                continue

            # TODO: Descomentar esta linea para probar como vende con esta estrategia
            # TODO: Probar diferentes valores de days en backesting (system2 = 20, system1 = 10)
            self.sell_like_turtletraders(position, days=50)


    # Usamos la técnica de venta de posición de los Turtle Traders
    def sell_like_turtletraders(self, position, days=20):
        self.log_message(f"Comprobando exit para {position.asset} según Turtle Traders")
        prices = self.get_historical_prices(position.asset, length=days).df
            
        if 'close' in prices.columns:
            # Calcular el mínimo de los últimos 'days' días
            min_days = prices['close'].tail(days).min()

            # Comprobar si el último precio es el mínimo de los últimos 'days' días
            latest_price = prices['close'].iloc[-1]
            if latest_price == min_days:
                self.log_message(f"{position.asset.symbol} has reached a new {days}-day low at {latest_price}, selling entire position.")
                # Crear y enviar una orden de mercado para vender toda la posición
                order = position.get_selling_order()
                self.submit_order(order)
            else:
                self.log_message(f"No sell condition met for {position.asset.symbol}")

        else:
            # Mensaje de error si no se encuentra la columna 'close'
            self.log_message(f"Error: No 'close' price available for {position.asset.symbol}")
    
    # Este método debe comprar de forma equitativa acciones de la lista de assets
    # hasta agotar el cash disponible, siempre comprando acciones enteras.
    def comprar_equitativamente(self):
        # Número de activos a comprar
        num_assets = len(self.assets)
        if num_assets == 0:
            return  # No hay activos para comprar

        # Calcular el monto máximo inicial para cada activo
        initial_allocation_per_asset = self.cash / num_assets

        # Este diccionario guardará cuántas acciones se compran de cada activo
        asset_purchases = {}

        for asset in self.assets:
            last_price = self.get_last_price(asset)
            if last_price <= 0:
                continue  # Evitar división por cero o precios no válidos

            # Calcular la cantidad máxima de acciones que se pueden comprar
            max_shares = int(initial_allocation_per_asset / last_price)
            if max_shares > 0:
                asset_purchases[asset] = max_shares

        # Calcular el costo total de las compras planificadas
        total_cost = sum(asset_purchases[asset] * self.get_last_price(asset) for asset in asset_purchases)

        # Si el costo total excede el efectivo disponible, ajustar compras
        while total_cost > self.cash and any(asset_purchases.values()):
            for asset in asset_purchases:
                if asset_purchases[asset] > 0:
                    asset_purchases[asset] -= 1  # Reducir en una acción
                    total_cost = sum(asset_purchases[asset] * self.get_last_price(asset) for asset in asset_purchases)
                    if total_cost <= self.cash:
                        break

        # Realizar las órdenes de compra
        for asset, quantity in asset_purchases.items():
            if quantity > 0:
                order = self.create_order(asset, quantity, "buy")
                self.submit_order(order)


    def get_all_symbols(self):
        return ["SPY", "GOOG", "TSLA", "AMZN"]

    def get_position_symbols(self):
        if self.is_backtesting:
            return self.get_all_symbols()
        
        # Obtenemos la lista de posiciones
        positions = self.get_positions()
        for position in positions:
            self.log_message(position)
            self.log_message(position.orders)
        
        # Usamos una comprensión de lista para extraer los símbolos
        symbols = [position.symbol for position in positions]
        
        return symbols
    

if __name__ == "__main__":
    is_live = False

    if is_live:
        from lumibot.brokers import Alpaca
        from config import ALPACA_CONFIG, TELEGRAM_CONFIG
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = MonitorStrategy(broker)

        # Set telegram bot and attach to strategy
        bot = TelegramBotHandler(TELEGRAM_CONFIG)
        strategy.set_messaging_bot(bot)

        # Set trader
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        from config import POLYGON_CONFIG
        backtesting_start = datetime.datetime(2024, 1, 1)
        backtesting_end = datetime.datetime(2024, 4, 21)
        trading_fee = TradingFee(percent_fee=0.001)

        trader = Trader(backtest=True)
        trading_fee = TradingFee(percent_fee=0.001)
        MonitorStrategy.backtest(
            PolygonDataBacktesting,
            backtesting_start,
            backtesting_end,
            polygon_api_key=POLYGON_CONFIG["KEY"],
            polygon_has_paid_subscription=POLYGON_CONFIG["PAID"],
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            benchmark_asset="SPY",
            risk_free_rate=0.532)