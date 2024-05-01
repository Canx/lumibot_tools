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
    
    def ma_crosses(self, symbol, short_length=50, long_length=200, ma_type='SMA', cross='bullish'):
        """
        Detecta cruces de dos medias móviles.

        Parameters:
        symbol (str): Símbolo del activo.
        short_length (int): Longitud de la media móvil de corto plazo.
        long_length (int): Longitud de la media móvil de largo plazo.
        ma_type (str): Tipo de media móvil, 'SMA' para media simple o 'EMA' para media exponencial.
        cross (str): Tipo de cruce, 'bullish' para corto sobre largo y 'bearish' para largo sobre corto.

        Returns:
        bool: True si se detecta el cruce especificado.
        """
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

            # Identificar el cruce
            latest_short_ma = short_ma.iloc[-1]
            latest_long_ma = long_ma.iloc[-1]
            previous_short_ma = short_ma.iloc[-2]
            previous_long_ma = long_ma.iloc[-2]

            if cross == 'bullish':
                if previous_short_ma < previous_long_ma and latest_short_ma > latest_long_ma:
                    self.log_message(f"{symbol}: Short-term MA ({short_length}) has crossed above Long-term MA ({long_length}).")
                    return True
            elif cross == 'bearish':
                if previous_short_ma > previous_long_ma and latest_short_ma < latest_long_ma:
                    self.log_message(f"{symbol}: Short-term MA ({short_length}) has crossed below Long-term MA ({long_length}).")
                    return True

        return False

    def ma_cross_with_atr_validation(self, symbol, short_length=50, long_length=200, ma_type='SMA', cross='bullish', atr_length=14, atr_factor=1):
        """
        Detecta cruces de dos medias móviles validados por el ATR para asegurar que la volatilidad respalda el movimiento.
        
        Parameters:
        symbol (str): Símbolo del activo.
        short_length (int): Longitud de la media móvil de corto plazo.
        long_length (int): Longitud de la media móvil de largo plazo.
        ma_type (str): Tipo de media móvil, 'SMA' o 'EMA'.
        cross (str): Tipo de cruce, 'bullish' para corto sobre largo y 'bearish' para largo sobre corto.
        atr_length (int): Período de cálculo del ATR.
        atr_factor (float): Factor para evaluar si el ATR actual es suficiente para validar el movimiento.

        Returns:
        bool: True si se detecta el cruce validado por el ATR.
        """
        # Verifica primero el cruce de medias móviles
        if self.ma_crosses(symbol, short_length, long_length, ma_type, cross):
            # Si hay cruce, calcula el ATR y valida
            atr = self.calculate_atr(symbol, atr_length)
            if atr is None:
                self.log_message(f"No ATR data available for {symbol}.")
                return False

            # Obtén el precio más reciente
            historical_prices = self.get_historical_prices(symbol, length=1)
            if historical_prices is None or 'close' not in historical_prices.df.columns:
                return False
            latest_price = historical_prices.df['close'].iloc[-1]

            # Verifica si el ATR es suficientemente alto para respaldar el cruce
            atr_threshold = latest_price * atr_factor / 100
            if atr >= atr_threshold:
                self.log_message(f"{symbol}: MA cross validated by ATR at {atr}, threshold was {atr_threshold}.")
                return True
            else:
                self.log_message(f"{symbol}: MA cross found but ATR at {atr} below threshold {atr_threshold}, not validated.")
                return False
        else:
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
    
    def rsi_vs_threshold(self, symbol, threshold, comparison='above'):
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
    
    def price_vs_bollinger(self, symbol, comparison='above', band='upper'):
        """
        Compara el precio actual de un activo con una de las bandas de Bollinger especificadas (superior, media o inferior) y determina si está por encima o por debajo de esta.
        """
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(symbol)
        latest_price = self.get_historical_prices(symbol, length=1).df['close'].iloc[-1]

        if band == 'upper':
            band_value = upper_band
            band_name = "upper band"
        elif band == 'middle':
            band_value = middle_band
            band_name = "middle band"
        elif band == 'lower':
            band_value = lower_band
            band_name = "lower band"
        else:
            self.log_message(f"Invalid band type specified: {band}. Choose 'upper', 'middle', or 'lower'.")
            return False

        if comparison == 'above' and latest_price > band_value:
            self.log_message(f"{symbol}: Price {latest_price} is above the {band_name}: {band_value}.")
            return True
        elif comparison == 'below' and latest_price < band_value:
            self.log_message(f"{symbol}: Price {latest_price} is below the {band_name}: {band_value}.")
            return True
        else:
            self.log_message(f"{symbol}: Price {latest_price} does not meet the condition of being {comparison} the {band_name} ({band_value}).")
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
        
    def order_block_signal(self, symbol, block_type, threshold_distance=0.01):
        """
        Genera señales de trading basadas en la proximidad a un Order Block específico (alcista o bajista).

        Parameters:
        symbol (str): Símbolo del activo para el cual generar la señal.
        block_type (str): Tipo de Order Block a detectar ('bullish' para alcista, 'bearish' para bajista).
        threshold_distance (float): Distancia máxima (en porcentaje del precio) para considerar que el precio está cerca del Order Block.

        Returns:
        bool: True si se detecta una señal relevante, False en caso contrario.
        """
        ohlc_with_blocks = self.calculate_order_blocks(symbol)
        if ohlc_with_blocks is None:
            return False  # No data available

        latest_price = ohlc_with_blocks['close'].iloc[-1]
        latest_block = ohlc_with_blocks.iloc[-1]

        if block_type == 'bullish' and latest_block['OrderBlockType'] == 'Bullish' and abs(latest_price - latest_block['OrderBlockLevel']) / latest_price <= threshold_distance:
            self.log_message(f"Buy signal for {symbol} at {latest_price} near bullish order block at {latest_block['OrderBlockLevel']}.")
            return True

        elif block_type == 'bearish' and latest_block['OrderBlockType'] == 'Bearish' and abs(latest_price - latest_block['OrderBlockLevel']) / latest_price <= threshold_distance:
            self.log_message(f"Sell signal for {symbol} at {latest_price} near bearish order block at {latest_block['OrderBlockLevel']}.")
            return True

        return False  # No relevant signal found

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
    
    def calculate_bollinger_bands(self, symbol, period=20, num_std=2):
        """
        Calcula las bandas de Bollinger para un activo específico.
        """
        historical_prices = self.get_historical_prices(symbol, length=period)
        if historical_prices is None or 'close' not in historical_prices.df.columns:
            self.log_message(f"No historical close prices available for {symbol}.")
            return None

        prices_df = historical_prices.df
        sma = prices_df['close'].rolling(window=period).mean()
        std_dev = prices_df['close'].rolling(window=period).std()

        upper_band = sma + (std_dev * num_std)
        lower_band = sma - (std_dev * num_std)

        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
    def calculate_order_blocks(self, ohlc: pd.DataFrame, lookback_period=3) -> pd.DataFrame:
        """
        Identifica los Order Blocks en una serie de datos de precios OHLC.

        Parameters:
        ohlc (pd.DataFrame): DataFrame de precios OHLC con columnas 'open', 'high', 'low', 'close', 'volume'.
        lookback_period (int): Número de velas anteriores a considerar para identificar un cambio significativo.

        Returns:
        pd.DataFrame: DataFrame con columnas 'OrderBlockType' y 'OrderBlockLevel'.
        """
        ohlc['OrderBlockType'] = np.nan
        ohlc['OrderBlockLevel'] = np.nan
        
        for i in range(lookback_period, len(ohlc)):
            if ohlc['close'].iloc[i] > ohlc['close'].iloc[i - lookback_period]:
                if ohlc['volume'].iloc[i] > ohlc['volume'].iloc[i - lookback_period] * 1.5:  # Aumento significativo del volumen
                    ohlc['OrderBlockType'].iloc[i] = 'Bullish'
                    ohlc['OrderBlockLevel'].iloc[i] = ohlc['low'].iloc[i]
            elif ohlc['close'].iloc[i] < ohlc['close'].iloc[i - lookback_period]:
                if ohlc['volume'].iloc[i] > ohlc['volume'].iloc[i - lookback_period] * 1.5:
                    ohlc['OrderBlockType'].iloc[i] = 'Bearish'
                    ohlc['OrderBlockLevel'].iloc[i] = ohlc['high'].iloc[i]

        return ohlc[['OrderBlockType', 'OrderBlockLevel']]