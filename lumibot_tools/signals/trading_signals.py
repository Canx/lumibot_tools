import numpy as np
import pandas as pd

class Signals:
    def __init__(self, strategy):
        self.strategy = strategy
        self.prices_cache = {}

    def get_historical_prices(self, *args, **kwargs):
        return self.strategy.get_historical_prices(*args, **kwargs)
    
    def log_message(self, *args, **kwargs):
        return self.strategy.log_message(*args, **kwargs)

    def new_price_high_or_low(self, symbol, days, type='high'):
        historical_prices = self.get_historical_prices(symbol, length=days)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]

            if type == 'high':
                # Excluyendo el último precio del cálculo del máximo
                extreme_value = prices_df['close'].iloc[:-1].max()  
                condition_met = latest_price > extreme_value  # Verifica que sea mayor que el máximo anterior
                message = "high"
            else:
                # Excluyendo el último precio del cálculo del mínimo
                extreme_value = prices_df['close'].iloc[:-1].min()  
                condition_met = latest_price < extreme_value  # Verifica que sea menor que el mínimo anterior
                message = "low"

            if condition_met:
                self.log_message(f"{symbol} has reached a new {days}-day {message} at {latest_price}.")
                return True
            else:
                self.log_message(f"No {message} condition met for {symbol}")
        else:
            # Mensaje de error si no se encuentra la columna 'close'
            self.log_message(f"Error: No 'close' price available for {symbol}")

        return False


    
    def price_crosses_MA(self, symbol, length=200, ma_type='SMA', cross_direction='up'):
        historical_prices = self.get_historical_prices(symbol, length=length+1)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            if ma_type == 'SMA':
                ma = prices_df['close'].rolling(window=length).mean()
            elif ma_type == 'EMA':
                ma = prices_df['close'].ewm(span=length, adjust=False).mean()
            else:
                self.log_message(f"Unsupported MA type: {ma_type}")
                return False

            latest_price = prices_df['close'].iloc[-1]
            previous_price = prices_df['close'].iloc[-2]
            latest_ma = ma.iloc[-1]
            previous_ma = ma.iloc[-2]

            if cross_direction == 'up':
                if previous_price < previous_ma and latest_price > latest_ma:
                    self.log_message(f"{symbol}: Price crossed above {ma_type} from below.")
                    return True
            elif cross_direction == 'down':
                if previous_price > previous_ma and latest_price < latest_ma:
                    self.log_message(f"{symbol}: Price crossed below {ma_type} from above.")
                    return True
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {symbol}.")

        return False
    
    def short_over_long_ma(self, symbol, short_length=50, long_length=200, ma_type='SMA'):
        historical_prices = self.get_historical_prices(symbol, length=max(short_length, long_length) + 1)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            if ma_type == 'SMA':
                short_ma = prices_df['close'].rolling(window=short_length).mean()
                long_ma = prices_df['close'].rolling(window=long_length).mean()
            elif ma_type == 'EMA':
                short_ma = prices_df['close'].ewm(span=short_length, adjust=False).mean()
                long_ma = prices_df['close'].ewm(span=long_length, adjust=False).mean()
            else:
                self.log_message(f"Unsupported MA type: {ma_type}")
                return False

            latest_short_ma = short_ma.iloc[-1]
            latest_long_ma = long_ma.iloc[-1]

            if latest_short_ma > latest_long_ma:
                self.log_message(f"{symbol}: Short-term MA ({short_length}) is above long-term MA ({long_length}).")
                return True
            else:
                self.log_message(f"{symbol}: Short-term MA ({short_length}) is below long-term MA ({long_length}).")
                return False
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {symbol}.")

        return False
    
    def rsi_above_or_below(self, symbol, threshold, comparison='above'):
        """
        Compara el RSI actual con un umbral dado y devuelve True si se cumple la condición.
        """
        rsi_value = self.calculate_rsi(symbol)
        if rsi_value is None:
            return False
        
        if comparison == 'above' and rsi_value > threshold:
            self.log_message(f"RSI for {symbol} is above {threshold}: {rsi_value}")
            return True
        elif comparison == 'below' and rsi_value < threshold:
            self.log_message(f"RSI for {symbol} is below {threshold}: {rsi_value}")
            return True
        else:
            self.log_message(f"RSI for {symbol} is {rsi_value}, does not meet the {comparison} than {threshold} condition.")
            return False
    

    ### trailing stops ###
    def trailing_stop_percent(self, symbol, trail_percent, lookback_period=90):
        # Obtener precios históricos para el símbolo
        historical_prices = self.get_historical_prices(symbol, length=lookback_period)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            peak_price = prices_df['close'].max()  # Calcular el máximo histórico de todos los precios disponibles

            # Calcular el precio de parada basado en el peak_price
            stop_price = peak_price * (1 - trail_percent / 100)

            # Verificar si el precio actual está por debajo del precio de parada
            if latest_price < stop_price:
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                return True
        return False
    
    def trailing_stop_atr(self, symbol, atr_multiplier=3, lookback_period=90):
        # Calcular ATR usando un periodo estándar de 14 días (o ajustar según sea necesario)
        atr = self.calculate_atr(symbol, 14)
        if atr is None:
            self.log_message(f"Unable to calculate ATR for {symbol}.")
            return False
        else:
            self.log_message(f"ATR for {symbol} calculated: {atr}")

        # Obtener precios históricos con el lookback_period para recalcular el peak_price
        historical_prices = self.get_historical_prices(symbol, length=lookback_period)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            peak_price = prices_df['close'].max()  # Recalcular el máximo histórico de todos los precios disponibles

            self.log_message(f"Calculated peak price for {symbol}: {peak_price}")

            # Calcular el precio de parada utilizando el ATR y el multiplicador
            stop_price = peak_price - atr * atr_multiplier
            self.log_message(f"Calculated trailing stop for {symbol}: {stop_price} (Peak price: {peak_price}, ATR: {atr}, Multiplier: {atr_multiplier})")

            # Verificar si el precio actual está por debajo del precio de parada
            if latest_price < stop_price:
                self.log_message(f"{symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                return True
            else:
                self.log_message(f"{symbol}: Price {latest_price} is still above the trailing stop {stop_price}. No sell action taken.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {symbol}.")
            return False

    #### Internal calculation methods ###
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
    
    def calculate_rsi(self, symbol, period=14):
        """
        Calcula el RSI (Índice de Fuerza Relativa) para un activo específico.
        """
        historical_prices = self.get_historical_prices(symbol, length=period+1)
        if historical_prices is None or 'close' not in historical_prices.df.columns:
            self.log_message(f"No historical close prices available for {symbol}.")
            return None
        
        prices_df = historical_prices.df
        close = prices_df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]