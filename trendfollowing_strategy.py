from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset
from lumibot.backtesting import PolygonDataBacktesting
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
from config import ALPACA_CONFIG, TELEGRAM_CONFIG, POLYGON_CONFIG
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import json
import os

"""
Estrategia de tipo trend following con fines educativos
"""
class TrendFollowingStrategy(MessagingStrategy):

    def initialize(self):
        self.sleeptime = "1D" if self.is_backtesting else "1H"
        self.max_assets = 10
        self.assets = self.get_start_assets() # TODO: get from persistence!
        self.update_assets()

    def get_backtesting_assets(self):
        #return ["GOOG", "TSLA", "AMZN", "NVDA", "AAPL", "MSFT", "META", "AMD", "WMT", "PEP", "LLY"]
        #return ["AMD", "WMT", "PEP", "LLY"]
        #return ["TSLA", "NVDA", "GOOG", "AMZN", "AAPL", "MSFT", "META"]
        return ["META"]
    
    # TODO: Use this method for filtering and getting new potential assets to enter.
    def after_market_closes(self):
        self.update_assets()

    def on_abrupt_closing(self):
        # Definir el nombre del archivo donde se guardarán los activos
        filename = 'assets.json'

        # Abrir el archivo para escribir
        with open(filename, 'w') as file:
            # Serializar la lista de activos y guardarla en el archivo
            json.dump(self.assets, file)

    def load_assets(self, filename='assets.json'):
        # Comprobar si el archivo existe
        if os.path.exists(filename):
            try:
                # Abrir el archivo y cargar los activos
                with open(filename, 'r') as file:
                    assets = json.load(file)
                return assets
            except json.JSONDecodeError:
                self.log_message("Error al decodificar el archivo de activos.")
            except Exception as e:
                self.log_message(f"Error al cargar el archivo de activos: {e}")
        else:
            self.log_message("No se encontró archivo de activos guardado.")

        return []

    def on_trading_iteration(self):
        entry_signal = lambda symbol: (
            self.signal_when_price_crossup_SMA(symbol, 200) or
            self.signal_when_new_high(symbol, 52*7)
        )

        exit_signal = lambda position: (
           #self.signal_sell_when_trailing_stop_atr(position, atr_multiplier=5)
           self.signal_when_trailing_stop_percent(position, 4)
        )

        self.check_exits(exit_signal)
        self.check_entries(entry_signal)

    
    def check_entries(self, signal_function):
        for symbol in self.assets:
            if signal_function(symbol):
                latest_price = self.get_last_price(symbol)
                shares_to_buy = self.calculate_shares_to_buy(symbol, latest_price)
                if shares_to_buy > 0:
                    order = self.create_order(symbol, shares_to_buy, "buy")
                    self.submit_order(order)
                    self.log_message(f"Placed order for {shares_to_buy} shares of {symbol} at price {latest_price}.")
                else:
                    self.log_message(f"Not enough cash to buy {symbol}.")

    def check_exits(self, signal_function):
        for position in self.get_positions():
            if position.asset.asset_type != Asset.AssetType.STOCK:
                self.log_message("It's not a stock!")
                continue

            if signal_function(position):
                self.log_message(f"Exiting position for {position.symbol}")
                if position.quantity > 0:
                    order = self.create_order(position.symbol, position.quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"Placed order to sell all {position.quantity} shares of {position.symbol}.")
                    if hasattr(position, 'peak_price'):
                        position.peak_price = None  # Resetear peak_price después de la venta
                else:
                    self.log_message(f"Attempted to place an order with negative quantity for {position.symbol}.")


    ########################
    ### CHATBOT COMMANDS ###
    ########################
    def add_symbol_command(self, symbol=None):
        if symbol is None:
            return "No symbol provided."

        # Intenta obtener el último precio para verificar si el símbolo existe en el broker
        last_price = self.get_last_price(symbol)
        if last_price is None:
            return f"Symbol {symbol} does not exist in the broker or an error occurred retrieving the price."

        if symbol not in self.assets:
            self.assets.append(symbol)
            return f"Symbol {symbol} added successfully."
        else:
            return f"Symbol {symbol} already exists in the list."


    def remove_symbol_command(self, symbol=None):
        if symbol is None:
            return "No symbol provided."

        # Verifica si el símbolo existe en la lista
        if symbol in self.assets:
            self.assets.remove(symbol)
            return f"Symbol {symbol} removed successfully."
        else:
            return f"Symbol {symbol} does not exist in the list."


    def list_symbols_command(self, parameters=None):
        return ', '.join(self.assets)

    #######################
    #### ENTRY SIGNALS ####
    #######################

    def signal_when_new_high(self, symbol, days):
        historical_prices = self.get_historical_prices(symbol, length=days)
        prices = historical_prices.df if historical_prices else None

        if prices is not None and 'close' in prices.columns:
            latest_price = prices['close'].iloc[-1]
            max_days = prices['close'].tail(days).max()
            if latest_price == max_days:
                self.log_message(f"{symbol} has reached a new {days}-day high at {latest_price}")
                return True
        return False
    
    def signal_when_price_crossup_SMA(self, symbol, days=200):
        historical_prices = self.get_historical_prices(symbol, length=days+1)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            sma = prices_df['close'].rolling(window=days).mean()
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]
            latest_sma = sma.iloc[-1]
            previous_sma = sma.iloc[-2]

            if previous_price < previous_sma and latest_price > latest_sma:
                self.log_message(f"{symbol}: Price crossed above SMA from below.")
                return True
        return False

    def signal_when_price_crossup_EMA(self, symbol, days=200):
        historical_prices = self.get_historical_prices(symbol, length=days+1)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            # Calculamos la EMA en lugar de la SMA
            ema = prices_df['close'].ewm(span=days, adjust=False).mean()
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]
            latest_ema = ema.iloc[-1]
            previous_ema = ema.iloc[-2]

            # Verificar el cruce del precio sobre la EMA
            if previous_price < previous_ema and latest_price > latest_ema:
                self.log_message(f"{symbol}: Price crossed above EMA from below.")
                return True
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {symbol}.")
        return False

    #######################
    #### EXIT SIGNALS #####
    #######################
    def signal_sell_when_trailing_stop_atr(self, position, atr_multiplier=3):
        symbol = position.symbol
        atr = self.calculate_atr(symbol, 14)
        if atr is None:
            self.log_message(f"Unable to calculate ATR for {symbol}.")
            return False
        else:
            self.log_message(f"ATR for {symbol} calculated: {atr}")

        historical_prices = self.get_historical_prices(symbol, length=2)  # Solo necesitamos los dos últimos precios
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]

            if hasattr(position, 'peak_price'):
                self.log_message(f"Current peak price for {symbol}: {position.peak_price}. New candidate price: {previous_price}")
                position.peak_price = max(position.peak_price, previous_price)
            else:
                self.log_message(f"Initializing peak price for {symbol} with price: {previous_price}")
                position.peak_price = previous_price
            
            stop_price = position.peak_price - atr * atr_multiplier
            self.log_message(f"Calculated trailing stop for {symbol}: {stop_price} (Peak price: {position.peak_price}, ATR: {atr}, Multiplier: {atr_multiplier})")

            if latest_price < stop_price:
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                return True
            else:
                self.log_message(f"{symbol}: Price {latest_price} is still above the trailing stop {stop_price}. No sell action taken.")
        
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")
            return False
    
    def signal_when_trailing_stop_percent(self, position, trail_percent):
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
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                return True
        return False

    def signal_when_price_crosses_down_SMA(self, position, days=200):
        symbol = position.symbol
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
                self.log_message(f"{symbol}: Price crossed below SMA from above.")
                return True
        return False

    def signal_sell_when_new_low(self, position, days):
        self.log_message(f"Comprobando exit para {position.asset} según Turtle Traders")
        historical_prices = self.get_historical_prices(position.asset, length=days)
        prices = historical_prices.df if historical_prices else None

        if prices is not None and 'close' in prices.columns:
            # Calcular el mínimo de los últimos 'days' días
            min_days = prices['close'].tail(days).min()

            # Comprobar si el último precio es el mínimo de los últimos 'days' días
            latest_price = prices['close'].iloc[-1]
            if latest_price == min_days:
                self.log_message(f"{position.asset.symbol} has reached a new {days}-day low at {latest_price}.")
                return True
            else:
                self.log_message(f"No sell condition met for {position.asset.symbol}")
        else:
            # Mensaje de error si no se encuentra la columna 'close'
            self.log_message(f"Error: No 'close' price available for {position.asset.symbol}")

        return False

    
    ##########################
    #### POSITION SIZING  ####
    ##########################
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
    ### FUNCIONES AUXILIARES ###
    ############################
    def calculate_number_of_eligible_assets(self):
        # Podría ajustarse para recalcular dinámicamente el número de activos elegibles en diferentes escenarios
        return len(self.assets)  # As an example, adjust based on actual eligibility logic
    
    def calculate_atr(self, symbol, period=14):
        historical_prices = self.get_historical_prices(symbol, length=period + 1)
        if historical_prices is None or 'high' not in historical_prices.df.columns or 'low' not in historical_prices.df.columns or 'close' not in historical_prices.df.columns:
            return None

        df = historical_prices.df
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        # Usar np.maximum.reduce para calcular el TR y luego convertirlo a una Serie de pandas
        tr = pd.Series(np.maximum.reduce([high_low, high_close, low_close]))

        # Calcular el ATR usando rolling y mean sobre la serie pandas
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

    ##########################
    ### ASSET MANAGEMENT  ####
    ##########################
    def update_assets(self):
        if not(self.is_backtesting):
            # Check if any position done outside bot
            position_assets = self.get_position_assets()
            for asset in position_assets:
                if asset not in self.assets:
                    self.assets.append(asset)

            # Add new assets from screener
            new_assets = self.get_assets_from_screener()
            for asset in new_assets:
                if asset not in self.assets:
                    self.assets.append(asset)

            # Filter best assets (max_assets)
            self.assets = self.filter_best_assets(self.assets)

    def get_assets_from_screener(self):
        from finvizfinance.screener.overview import Overview

        foverview = Overview()
        filters_dict = {
            'Current Volume': 'Over 1M',
            'Beta': 'Over 1',
            'Performance': 'Month +10%',
            'Performance 2': 'Year +30%',
            'RSI (14)': 'Not Overbought (<60)',
            #'200-Day Simple Moving Average': 'Price bellow SMA200',
            '50-Day Simple Moving Average': 'Price above SMA50',
            'Gap': 'Up'
        }
        foverview.set_filter(filters_dict=filters_dict)
        df = foverview.screener_view()

        # Verificar si el DataFrame está vacío
        if df is not None and not df.empty:
            # Si el DataFrame no está vacío, extraer la lista de tickers
            symbol_list = df['Ticker'].tolist()
            self.log_message(f"Símbolos detectados por el screener: {symbol_list}")
        else:
            # Si el DataFrame está vacío, manejar adecuadamente
            symbol_list = []
            print("No se encontraron símbolos en el screener.")
        
        # Quitamos duplicados manteniendo el orden
        symbol_list = list(dict.fromkeys(symbol_list))
        
        return symbol_list

    # Filter the best max_assets based on momentum ratio
    def filter_best_assets(self, assets):
        asset_scores = []

        for asset in assets:
            # Obtener datos históricos
            historical_prices = self.get_historical_prices(asset, length=201)  # Asumiendo que esta función devuelve un DataFrame
            data = historical_prices.df


            # Calcular SMA de 50 días y SMA de 200 días
            data['SMA50'] = data['close'].rolling(window=50).mean()
            data['SMA200'] = data['close'].rolling(window=200).mean()

            # Evaluar la condición de cruce para determinar la puntuación
            latest_sma50 = data['SMA50'].iloc[-1]
            latest_sma200 = data['SMA200'].iloc[-1]
            score = latest_sma50 / latest_sma200  # Un ratio simple para la puntuación

            asset_scores.append((asset, score))

        # Ordenar todos los activos por su puntuación, los más altos primero
        sorted_assets = sorted(asset_scores, key=lambda x: x[1], reverse=True)

        # Si deseas asegurarte de tener al menos `self.max_assets`, se seleccionan aquí
        best_assets = [asset[0] for asset in sorted_assets[:max(len(sorted_assets), self.max_assets)]]

        return best_assets

    def get_start_assets(self):
        if self.is_backtesting:
            return self.get_backtesting_assets()
        
        else:
            # Cargar activos guardados desde un archivo
            return self.load_assets()

    def get_position_assets(self):
        # Obtenemos la lista de posiciones
        positions = self.get_positions()
        for position in positions:
            self.log_message(position)
            self.log_message(position.orders)
            
        # Usamos una comprensión de lista para extraer los símbolos
        assets = [position.symbol for position in positions]
            
        return assets
    

if __name__ == "__main__":
    is_live = True

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
        backtesting_start = datetime.datetime(2023, 4, 1)
        backtesting_end = datetime.datetime(2024, 4, 1)
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
