from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset
from lumibot.backtesting import PolygonDataBacktesting
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
from config import ALPACA_CONFIG, TELEGRAM_CONFIG, POLYGON_CONFIG
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

"""
Estrategia de tipo trend following
"""
class TrendFollowingStrategy(MessagingStrategy):

    def initialize(self):
        self.sleeptime = "1D"
        self.assets = self.get_position_symbols()

    def on_trading_iteration(self):
        self.check_exits()
        self.check_entries()

    def check_entries(self):
        all_symbols = self.get_all_symbols()
        
        current_positions = {position.symbol: position for position in self.get_positions()}
        unowned_or_zero_assets = [symbol for symbol in all_symbols if symbol not in current_positions or current_positions[symbol].quantity == 0]

        for symbol in unowned_or_zero_assets:
            self.buy_when_price_crosses_SMA(symbol, days=200)
            self.buy_when_new_high(symbol, 52*7)
    

    def check_exits(self):
        for position in self.get_positions():
            asset = position.asset

            self.log_message("Asset:" + asset.symbol)
            if asset.asset_type != Asset.AssetType.STOCK:
                self.log_message("No es stock!")
                continue
            
            #self.sell_when_price_crosses_SMA(position, days=50)

            self.sell_when_trailing_stop_percent(position, 4) # sell when trailing stop arrives at 4%
            #self.sell_when_trailing_stop_atr(position, 3)
            #self.sell_when_new_low(position, days=50)

    
    ############################
    #### SEÑALES DE COMPRA #####
    ############################
    def buy_when_new_high(self, symbol, days):
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
                shares_to_buy = self.calculate_shares_to_buy(symbol, latest_price)

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

    def buy_when_price_crosses_SMA(self, symbol, days=200):
        # Obtener los precios históricos para el símbolo dado
        historical_prices = self.get_historical_prices(symbol, length=days+1)  # Añadimos 1 día más para obtener el precio anterior
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            # Calculamos el SMA para los días especificados
            sma = prices_df['close'].rolling(window=days).mean()

            # Obtenemos el último precio de cierre y el precio de cierre del día anterior
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]
            latest_sma = sma.iloc[-1]
            previous_sma = sma.iloc[-2]

            # Comprobamos que el precio anterior esté por debajo del SMA y que el último precio esté por encima del SMA
            if previous_price < previous_sma and latest_price > latest_sma:
                self.log_message(f"{symbol}: Price crossed above SMA from below, preparing to buy.")
                shares_to_buy = self.calculate_shares_to_buy(symbol, latest_price)

                if shares_to_buy > 0:
                    order = self.create_order(symbol, shares_to_buy, "buy")
                    self.submit_order(order)
                    self.log_message(f"Placed order for {shares_to_buy} shares of {symbol} at price {latest_price}.")
                else:
                    self.log_message(f"Not enough cash to buy {symbol}.")
            else:
                self.log_message(f"No buy signal for {symbol} as price has not crossed SMA from below.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")

    ############################
    #### TAMAÑO DE POSICION ####
    ############################

    # Función pública
    def calculate_shares_to_buy(self, symbol, price_per_share):
        return self.calculate_shares_to_buy_basic(symbol, price_per_share)

    def calculate_shares_to_buy_basic(self, symbol, price_per_share):
        if price_per_share <= 0:
            return 0
        available_cash = self.cash / self.calculate_number_of_eligible_assets()
        shares_to_buy = int(available_cash / price_per_share)
        return shares_to_buy

    def calculate_shares_to_buy_var(self, symbol, price_per_share):
        if price_per_share <= 0:
            return 0
        
        # Calcular el VaR para el símbolo
        VaR = self.calculate_var(symbol)
        self.log_message(f"VaR para {symbol}: {VaR}")
        
        # Supongamos que no queremos arriesgar más del 2% del capital en una sola operación
        max_risk_per_trade = 0.02 * self.cash
        self.log_message("max_risk_per_trade:{max_risk_per_trade}")
        # Calcular el monto máximo que podemos perder basado en el VaR
        max_loss = VaR * price_per_share
        self.log_message(f"max_loss: {max_loss}")
        
        # Calcular cuánto capital asignar a esta compra
        capital_to_use = min(max_loss, max_risk_per_trade)
        shares_to_buy = int(capital_to_use / price_per_share)
        
        return shares_to_buy

    def calculate_var(self, symbol, confidence_level=0.95):
        # Aquí necesitas obtener los precios históricos para calcular la volatilidad
        historical_prices = self.get_historical_prices(symbol, length=252)  # Ejemplo: últimos 252 días hábiles (1 año)
        if historical_prices is None or 'close' not in historical_prices.df.columns:
            return 0
        
        # Calcular los retornos logarítmicos diarios
        returns = np.log(historical_prices.df['close'] / historical_prices.df['close'].shift(1))
        # Estimar la volatilidad como la desviación estándar de los retornos
        sigma = returns.std()
        # Calcular el VaR diario a un nivel de confianza dado
        VaR = sigma * norm.ppf(confidence_level)
        return VaR

    



    ############################
    #### SEÑALES DE VENTA  #####
    ############################
    def sell_when_price_crosses_SMA(self, position, days=200):
        symbol = position.symbol

        # Obtener los precios históricos para el símbolo dado
        historical_prices = self.get_historical_prices(symbol, length=days+1)  # Añadimos 1 día más para obtener el precio anterior
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            # Calculamos el SMA para los días especificados
            sma = prices_df['close'].rolling(window=days).mean()

            # Obtenemos el último precio de cierre y el precio de cierre del día anterior
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]
            latest_sma = sma.iloc[-1]
            previous_sma = sma.iloc[-2]

            # Comprobamos que el precio anterior esté por encima del SMA y que el último precio esté por debajo del SMA
            if previous_price > previous_sma and latest_price < latest_sma:
                self.log_message(f"{symbol}: Price crossed below SMA from above, preparing to sell.")
                # Obtener la posición actual para saber cuántas acciones vender
                if position and position.quantity > 0:
                    order = self.create_order(symbol, position.quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"Placed order to sell all {position.quantity} shares of {symbol} at price {latest_price}.")
                else:
                    self.log_message(f"No position to sell for {symbol}.")
            else:
                self.log_message(f"No sell signal for {symbol} as price has not crossed SMA from above.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")

    # Usamos la técnica de venta de posición de los Turtle Traders
    def sell_when_new_low(self, position, days):
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

    def sell_when_trailing_stop(self, position):
        #self.sell_when_trailing_stop_percent(position, 4)
        self.sell_when_trailing_stop_atr(position, 3)

    def sell_when_trailing_stop_percent(self, position, trail_percent):
        symbol = position.symbol
        historical_prices = self.get_historical_prices(symbol, length=2)  # Solo necesitamos los dos últimos precios
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]

            if hasattr(position, 'peak_price'):
                position.peak_price = max(position.peak_price, previous_price)
            else:
                position.peak_price = previous_price  # Inicializar peak_price si no está presente

            stop_price = position.peak_price * (1 - trail_percent / 100)

            if latest_price < stop_price:
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}, preparing to sell.")
                if position.quantity > 0:
                    order = self.create_order(symbol, position.quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"Placed order to sell all {position.quantity} shares of {symbol} at price {latest_price}.")
                    position.peak_price = None  # Resetear peak_price después de la venta
                else:
                    self.log_message(f"No position to sell for {symbol}.")
            else:
                self.log_message(f"{symbol}: Price is still above the trailing stop. No sell action taken.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")

    def sell_when_trailing_stop_atr(self, position, atr_multiplier=3):
        symbol = position.symbol
        atr = self.calculate_atr(symbol)
        if atr is None:
            self.log_message(f"Unable to calculate ATR for {symbol}.")
            return

        historical_prices = self.get_historical_prices(symbol, length=2)  # Solo necesitamos los dos últimos precios
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]

            if hasattr(position, 'peak_price'):
                position.peak_price = max(position.peak_price, previous_price)
            else:
                position.peak_price = previous_price  # Inicializar peak_price si no está presente

            stop_price = position.peak_price - atr * atr_multiplier

            if latest_price < stop_price:
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}, preparing to sell.")
                if position.quantity > 0:
                    order = self.create_order(symbol, position.quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"Placed order to sell all {position.quantity} shares of {symbol} at price {latest_price}.")
                    position.peak_price = None  # Resetear peak_price después de la venta
                else:
                    self.log_message(f"No position to sell for {symbol}.")
            else:
                self.log_message(f"{symbol}: Price is still above the trailing stop. No sell action taken.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")
    ############################
    ### FUNCIONES AUXILIARES ###
    ############################
    def calculate_number_of_eligible_assets(self):
        # Podría ajustarse para recalcular dinámicamente el número de activos elegibles en diferentes escenarios
        return len(self.get_all_symbols())  # As an example, adjust based on actual eligibility logic
    
    def calculate_atr(self, symbol, period=14):
        historical_prices = self.get_historical_prices(symbol, length=period + 1)
        if historical_prices is None or 'high' not in historical_prices.df.columns or 'low' not in historical_prices.df.columns or 'close' not in historical_prices.df.columns:
            return None

        # Asegúrate de que los cálculos se realizan sobre un DataFrame de pandas
        df = historical_prices.df
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        # Combine all three into a DataFrame and take the max
        tr = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)

        # Calcula el ATR usando rolling y mean sobre la serie pandas
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr
    
    ####################################
    #### FUNCIONES DE COMPRA INICIAL ###
    ####################################
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

    # Esta función intenta ajustar las compras iniciales según la volatilidad de los activos.
    def comprar_segun_volatilidad(self):
        pass


    def get_all_symbols(self):
        return ["GOOG", "TSLA", "AMZN", "NVDA", "AAPL", "MSFT", "META"]

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
        strategy = TrendFollowingStrategy(broker)

        # Set telegram bot and attach to strategy
        bot = TelegramBotHandler(TELEGRAM_CONFIG)
        strategy.set_messaging_bot(bot)

        # Set trader
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        from config import POLYGON_CONFIG
        backtesting_start = datetime.datetime(2023, 1, 1)
        backtesting_end = datetime.datetime(2023, 12, 31)
        trading_fee = TradingFee(percent_fee=0.001)

        trader = Trader(backtest=True)
        trading_fee = TradingFee(percent_fee=0.001)
        TrendFollowingStrategy.backtest(
            PolygonDataBacktesting,
            backtesting_start,
            backtesting_end,
            polygon_api_key=POLYGON_CONFIG["KEY"],
            polygon_has_paid_subscription=POLYGON_CONFIG["PAID"],
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            benchmark_asset="SPY",
            risk_free_rate=0.0532)