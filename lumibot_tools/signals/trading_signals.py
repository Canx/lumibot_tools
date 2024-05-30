import math
import numpy as np
import pandas as pd
import pandas_ta as ta
from lumibot.entities.order import Order

class Signals:
    def __init__(self, strategy):
        self.strategy = strategy
        self._multiplier = None
        self.signal_history = {}

    @property
    def multiplier(self):
        if self._multiplier is None:
            self._multiplier = self.get_multiplier()
        
        return self._multiplier
    
    def get_datetime(self):
        return self.strategy.broker.datetime
    
    def get_sleeptime(self):
        return self.strategy.sleeptime
    
    def get_timestep(self):
        return self.strategy.broker.data_source.get_timestep()
    
    def get_historical_prices(self, *args, **kwargs):
        length = kwargs.get('length', 1)
        length = length * self.multiplier
        kwargs['length'] = length
        return self.strategy.get_historical_prices(timestep = self.get_timestep(), *args, **kwargs)
    
    def log_message(self, *args, **kwargs):
        return self.strategy.log_message(*args, **kwargs)
    
    def get_position(self, *args, **kwargs):
        return self.strategy.get_position(*args, **kwargs)
    
    def get_last_price(self, *args, **kwargs):
        return self.strategy.get_last_price(*args, **kwargs)
    
    def get_timestep(self):
        return self.strategy.broker.data_source.get_timestep()
    
    ## Cache methods ##
    def generate_cache_key(self, method_name, symbol, current_date, *args):
        return (method_name, symbol, current_date) + args
    
    def filter_signals(self, symbol, start_date, **kwargs):
        """
        Filters the signal history for a given symbol, start date, and additional parameters.

        Parameters:
        symbol (str): The symbol of the asset.
        start_date (datetime): The start date to filter signals from.
        **kwargs: Additional parameters to filter signals.

        Returns:
        List[Tuple]: A list of tuples containing the filtered signals.
        """
        filtered_signals = [
            (method, sym, dt, *params, sig) for (method, sym, dt, *params), sig in self.signal_history.items()
            if sym == symbol and dt >= start_date.date() and all(kwargs[key] == params[i] for i, key in enumerate(kwargs))
        ]
        return filtered_signals
    
    # Cache aware signal
    def price_crosses_MA(self, asset, length=200, ma_type='SMA', cross_direction='up'):
        """
        Determines if the price of a given symbol has crossed its moving average in a specified direction.

        Parameters:
        asset (Asset): The Asset class price_crosses_MA the asset.
        length (int): The length of the moving average.
        ma_type (str): Type of moving average, either 'SMA' (Simple Moving Average) or 'EMA' (Exponential Moving Average).
        cross_direction (str): The direction of the cross, either 'up' for upward or 'down' for downward.

        Returns:
        bool: True if the price crosses the moving average as specified, False otherwise.
        """
        current_date = self.get_datetime().date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key("price_crosses_MA", asset.symbol, current_date, length, ma_type, cross_direction)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]
        
        historical_prices = self.get_historical_prices(asset, length=length+1)
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

            signal_triggered = False

            if cross_direction == 'up':
                if previous_price < previous_ma and latest_price > latest_ma:
                    self.log_message(f"{asset.symbol}: Price crossed above {ma_type} from below.")
                    signal_triggered = True
            elif cross_direction == 'down':
                if previous_price > previous_ma and latest_price < latest_ma:
                    self.log_message(f"{asset.symbol}: Price crossed below {ma_type} from above.")
                    signal_triggered = True

            # Update signal history with datetime
            self.signal_history[cache_key] = signal_triggered

            return signal_triggered
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {asset.symbol}.")

        return False
    
    # Cache aware
    def new_price_high_or_low(self, asset, length, type='high'):
        """
        Checks if the latest price for a given symbol is either a new high or low over a specified number of days.

        Parameters:
        asset (Asset): The Asset class for the asset.
        length (int): The number of timesteps over which to check for new highs or lows.
        type (str, optional): Specifies whether to check for a new 'high' or 'low'. Defaults to 'high'.

        Returns:
        bool: True if a new high or low is detected, False otherwise.
        """

        current_date = self.get_datetime().date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key("new_price_high_or_low", asset.symbol, current_date, length, type)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        historical_prices = self.get_historical_prices(asset, length=length)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]

            if type == 'high':
                extreme_value = prices_df['close'].iloc[:-1].max()
                condition_met = latest_price > extreme_value
                message = "high"
            else:
                extreme_value = prices_df['close'].iloc[:-1].min()
                condition_met = latest_price < extreme_value
                message = "low"

            if condition_met:
                self.log_message(f"{asset.symbol} has reached a new {length}-{self.get_timestep()} {message} at {latest_price}.")
            else:
                self.log_message(f"No {message} condition met for {asset.symbol}")

            # Update signal history with datetime
            self.signal_history[cache_key] = condition_met

            return condition_met
        else:
            self.log_message(f"Error: No 'close' price available for {asset.symbol}")

        return False

    # cache aware
    def price_above_below_EMA(self, asset, length=200, position='above'):
        """
        Determines if the price of a given asset is above or below its exponential moving average (EMA).

        Parameters:
        asset (Asset): The Asset class for the asset.
        length (int): The length of the moving average.
        position (str): The expected position of the price relative to the EMA, either 'above' or 'below'.

        Returns:
        bool: True if the price is in the specified position relative to the EMA, False otherwise.
        """
        method_name = "price_above_below_EMA"
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, length, position)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        historical_prices = self.get_historical_prices(asset, length=length)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            ema = prices_df['close'].ewm(span=length, adjust=False).mean()
            latest_price = prices_df['close'].iloc[-1]
            latest_ema = ema.iloc[-1]

            if position == 'above' and latest_price > latest_ema:
                self.log_message(f"{asset.symbol}: Price is above the EMA.")
                condition_met = True
            elif position == 'below' and latest_price < latest_ema:
                self.log_message(f"{asset.symbol}: Price is below the EMA.")
                condition_met = True
            else:
                condition_met = False

            # Update signal history with datetime
            self.signal_history[cache_key] = condition_met

            return condition_met
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {asset.symbol}.")
            return False
    
    def ma_crosses(self, asset, short_length=50, long_length=200, ma_type='SMA', cross='bullish'):
        """
        Detects when a short-term moving average crosses a long-term moving average.

        Parameters:
        asset (Asset): The Asset class for the asset.
        short_length (int): The period of the short-term moving average.
        long_length (int): The period of the long-term moving average.
        ma_type (str): Type of moving average, 'SMA' for simple or 'EMA' for exponential.
        cross (str): Type of cross, 'bullish' for short-term crossing above long-term, 'bearish' for short-term crossing below long-term.

        Returns:
        bool: True if the specified cross is detected, False otherwise.
        """
        method_name = "ma_crosses"
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, short_length, long_length, ma_type, cross)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        historical_prices = self.get_historical_prices(asset, length=max(short_length, long_length) + 1)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            if 'datetime' in prices_df.columns:
                prices_df.set_index('datetime', inplace=True)

            if ma_type == 'SMA':
                prices_df['short_ma'] = ta.sma(prices_df['close'], length=short_length)
                prices_df['long_ma'] = ta.sma(prices_df['close'], length=long_length)
            elif ma_type == 'EMA':
                prices_df['short_ma'] = ta.ema(prices_df['close'], length=short_length)
                prices_df['long_ma'] = ta.ema(prices_df['close'], length=long_length)
            else:
                self.log_message(f"Unsupported MA type: {ma_type}")
                return False

            if cross == 'bullish':
                condition = (prices_df['short_ma'].shift(1) < prices_df['long_ma'].shift(1)) & (prices_df['short_ma'] > prices_df['long_ma'])
            else:
                condition = (prices_df['short_ma'].shift(1) > prices_df['long_ma'].shift(1)) & (prices_df['short_ma'] < prices_df['long_ma'])

            condition_met = condition.iloc[-1]
            if condition_met:
                self.log_message(f"{asset.symbol}: Short-term MA ({short_length}) has crossed {'above' if cross == 'bullish' else 'below'} Long-term MA ({long_length}).")

            # Update signal history with datetime
            self.signal_history[cache_key] = condition_met

            return condition_met

        return False

    # Cache aware
    def short_over_long_ma(self, asset, short_length=50, long_length=200, ma_type='SMA'):
        """
        Determines if the short-term moving average is above the long-term moving average.

        Parameters:
        asset (Asset): The Asset class for the asset.
        short_length (int): The period of the short-term moving average.
        long_length (int): The period of the long-term moving average.
        ma_type (str): Type of moving average, 'SMA' for simple or 'EMA' for exponential.

        Returns:
        bool: True if the short-term moving average is above the long-term moving average, False otherwise.
        """
        method_name = "short_over_long_ma"
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, short_length, long_length, ma_type)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        historical_prices = self.get_historical_prices(asset, length=max(short_length, long_length) + 1)
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

            condition_met = latest_short_ma > latest_long_ma
            if condition_met:
                self.log_message(f"{asset.symbol}: Short-term MA ({short_length}) is above long-term MA ({long_length}).")
            else:
                self.log_message(f"{asset.symbol}: Short-term MA ({short_length}) is below long-term MA ({long_length}).")

            # Update signal history with datetime
            self.signal_history[cache_key] = condition_met

            return condition_met
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {asset.symbol}.")

        return False
    
    # Cache aware
    def macd_signal(self, asset, fast_length=12, slow_length=26, signal_length=9, entry_type='bullish'):
        """
        Determines if a MACD line crosses its signal line, indicating potential buy or sell signals.

        Parameters:
        asset (Asset): The Asset class for the asset.
        fast_length (int): The period of the fast EMA.
        slow_length (int): The period of the slow EMA.
        signal_length (int): The period of the signal line.
        entry_type (str): Type of entry signal, 'bullish' for MACD crossing above signal line, 'bearish' for MACD crossing below signal line.

        Returns:
        bool: True if the specified MACD crossover is detected, False otherwise.
        """
        method_name = "macd_signal"
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, fast_length, slow_length, signal_length, entry_type)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        historical_prices = self.get_historical_prices(asset, length=max(fast_length, slow_length, signal_length) + signal_length)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            # Calculate MACD using the pandas_ta library
            macd = ta.macd(prices_df['close'], fast=fast_length, slow=slow_length, signal=signal_length)

            # Dynamic column names based on parameters
            macd_line_col = f'MACD_{fast_length}_{slow_length}_{signal_length}'
            signal_line_col = f'MACDs_{fast_length}_{slow_length}_{signal_length}'
            macd_histogram_col = f'MACDh_{fast_length}_{slow_length}_{signal_length}'

            if macd_line_col in macd.columns and signal_line_col in macd.columns:
                macd_line = macd[macd_line_col]
                signal_line = macd[signal_line_col]

                if entry_type == 'bullish':
                    condition = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
                elif entry_type == 'bearish':
                    condition = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)
                else:
                    self.log_message(f"Unsupported entry type: {entry_type}")
                    return False

                condition_met = condition.iloc[-1]
                if condition_met:
                    self.log_message(f"{asset.symbol}: MACD line has crossed {'above' if entry_type == 'bullish' else 'below'} the signal line.")

                # Update signal history with datetime
                self.signal_history[cache_key] = condition_met

                return condition_met
            else:
                self.log_message(f"Error: MACD calculation failed or column names are incorrect for {asset.symbol}")
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns:
                self.log_message(f"'Close' column missing for {asset.symbol}.")

        return False


    # Cache aware
    def rsi_vs_threshold(self, asset, threshold, comparison='above'):
        """
        Compares the current Relative Strength Index (RSI) of a symbol with a given threshold and returns True if the condition is met.

        Parameters:
        asset (Asset): The Asset class for the asset.
        threshold (float): The threshold to compare against.
        comparison (str): Condition to test ('above' or 'below').

        Returns:
        bool: True if the RSI meets the condition specified, False otherwise.
        """
        method_name = "rsi_vs_threshold"
        current_datetime = self.get_current_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, threshold, comparison)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        rsi_value = self.calculate_rsi(asset)
        if rsi_value is None:
            return False
        
        if comparison == 'above' and rsi_value > threshold:
            self.log_message(f"RSI for {asset.symbol} is above {threshold}: {rsi_value}")
            condition_met = True
        elif comparison == 'below' and rsi_value < threshold:
            self.log_message(f"RSI for {asset.symbol} is below {threshold}: {rsi_value}")
            condition_met = True
        else:
            self.log_message(f"RSI for {asset.symbol} is {rsi_value}, does not meet the {comparison} than {threshold} condition.")
            condition_met = False

        # Update signal history with datetime
        self.signal_history[cache_key] = condition_met

        return condition_met
    
    # Cache aware
    def price_vs_bollinger(self, asset, comparison='above', band='upper'):
        """
        Compares the current price of an asset against a specified Bollinger Band (upper, middle, or lower) and determines if it is above or below it.

        Parameters:
        asset (Asset): The Asset class for the asset.
        comparison (str): Specifies whether to check if the price is 'above' or 'below' the band.
        band (str): Specifies which Bollinger Band to compare against ('upper', 'middle', 'lower').

        Returns:
        bool: True if the price meets the condition specified relative to the Bollinger Band, False otherwise.
        """
        method_name = "price_vs_bollinger"
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()

        # Generate a unique key for the signal
        cache_key = self.generate_cache_key(method_name, asset.symbol, current_date, comparison, band)

        # Check if the signal already exists for the current date
        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(asset)
        latest_price = self.get_historical_prices(asset, length=1).df['close'].iloc[-1]

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
            self.log_message(f"{asset.symbol}: Price {latest_price} is above the {band_name}: {band_value}.")
            condition_met = True
        elif comparison == 'below' and latest_price < band_value:
            self.log_message(f"{asset.symbol}: Price {latest_price} is below the {band_name}: {band_value}.")
            condition_met = True
        else:
            self.log_message(f"{asset.symbol}: Price {latest_price} does not meet the condition of being {comparison} the {band_name} ({band_value}).")
            condition_met = False

        # Update signal history with datetime
        self.signal_history[cache_key] = condition_met

        return condition_met


    def trailing_stop_percent(self, asset, trail_percent):
        """
        Implements a dynamic trailing stop based on the maximum price reached during the day.
        The trailing stop adjusts based on the daily high price and is reset once triggered.

        Parameters:
        asset (Asset): The asset for which the trailing stop is being calculated.
        trail_percent (float): The percentage below the peak price at which the trailing stop is set.

        Returns:
        bool: True if the current price has fallen below the trailing stop price, indicating a sell signal. False otherwise.
        """
        try:
            # Retrieve the latest historical price data, asking for just the most recent day
            latest_price = self.get_last_price(asset)
            historical_prices = self.get_historical_prices(asset, length=1)
            if historical_prices is None or 'high' not in historical_prices.df.columns:
                self.log_message("No valid high price data available for trailing stop calculation.")
                return False

            # Access the high price from the historical data
            daily_high_price = historical_prices.df['high'].max()

            position = self.get_position(asset)
            if position is None:
                self.log_message("No position found for this asset.")
                return False

            # Ensure the peak_price attribute exists in position, initialize if not
            if not hasattr(position, 'peak_price'):
                position.peak_price = None
            
            # Find the last buy order that was filled
            last_buy_price = None
            if position.orders:
                for order in reversed(position.orders):  # Iterate in reverse to find the most recent
                    if order.side == 'buy' and order.is_filled():
                        last_buy_price = order.get_fill_price()
                        break

            # Determine the appropriate peak price
            potential_new_peak = max(filter(None, [daily_high_price, last_buy_price]))
            if position.peak_price is None or potential_new_peak > position.peak_price:
                position.peak_price = potential_new_peak
                self.log_message(f"Updated peak_price for {asset.symbol} to {position.peak_price} based on the daily high, or last buy order price.")

            # Use the last close price to compare with the trailing stop price
            stop_price = position.peak_price * (1 - trail_percent / 100)

            # Check if the current price is below the trailing stop price
            if latest_price < stop_price:
                self.log_message(f"{asset.symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                position.peak_price = None  # Reset the peak price after triggering the trailing stop
                return True

        except Exception as e:
            self.log_message(f"An error occurred: {str(e)}")
            return False

        
    
    def trailing_stop_atr(self, asset, atr_multiplier=3, lookback_period=90):
        """
        Implements a trailing stop based on the Average True Range (ATR), adjusted by a specified multiplier.

        Parameters:
        asset (Asset):  The Asset class for the asset.
        atr_multiplier (float): The factor by which the ATR is multiplied to set the stop price.
        lookback_period (int): The number of days to look back to determine the peak price and calculate the ATR.

        Returns:
        bool: True if the current price has fallen below the trailing stop price, False otherwise.
        """

        atr = self.calculate_atr(asset, 14)
        if atr is None:
            self.log_message(f"Unable to calculate ATR for {asset.symbol}.")
            return False
        else:
            self.log_message(f"ATR for {asset.symbol} calculated: {atr}")

        historical_prices = self.get_historical_prices(asset, length=lookback_period)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            latest_price = prices_df['close'].iloc[-1]
            peak_price = prices_df['close'].max()

            self.log_message(f"Calculated peak price for {asset.symbol}: {peak_price}")

            stop_price = peak_price - atr * atr_multiplier
            self.log_message(f"Calculated trailing stop for {asset.symbol}: {stop_price} (Peak price: {peak_price}, ATR: {atr}, Multiplier: {atr_multiplier})")

            if latest_price < stop_price:
                self.log_message(f"{asset.symbol}: Price {latest_price} has fallen below the trailing stop {stop_price}.")
                return True
            else:
                self.log_message(f"{asset.symbol}: Price {latest_price} is still above the trailing stop {stop_price}. No sell action taken.")
        else:
            self.log_message(f"Unable to retrieve prices or 'close' column missing for {asset.symbol}.")
            return False
        
    def order_block_signal(self, asset, block_type, threshold_distance=0.01):
        """
        Generates trading signals based on the proximity to a specific order block type (bullish or bearish).

        Parameters:
        asset (Asset):  The Asset class for the asset.
        block_type (str): The type of order block to detect ('bullish' for buying opportunities, 'bearish' for selling opportunities).
        threshold_distance (float): The maximum percentage distance from the order block level to consider the price to be near the order block.

        Returns:
        bool: True if a relevant signal is detected near the order block, False otherwise.
        """

        ohlc_with_blocks = self.calculate_order_blocks(asset)
        if ohlc_with_blocks is None:
            return False

        latest_price = ohlc_with_blocks['close'].iloc[-1]
        latest_block = ohlc_with_blocks.iloc[-1]

        if block_type == 'bullish' and latest_block['OrderBlockType'] == 'Bullish' and abs(latest_price - latest_block['OrderBlockLevel']) / latest_price <= threshold_distance:
            self.log_message(f"Buy signal for {asset.symbol} at {latest_price} near bullish order block at {latest_block['OrderBlockLevel']}.")
            return True

        elif block_type == 'bearish' and latest_block['OrderBlockType'] == 'Bearish' and abs(latest_price - latest_block['OrderBlockLevel']) / latest_price <= threshold_distance:
            self.log_message(f"Sell signal for {asset.symbol} at {latest_price} near bearish order block at {latest_block['OrderBlockLevel']}.")
            return True

        return False


    def calculate_atr(self, asset, period=14):
        """
        Calculates the Average True Range (ATR) for a given asset over a specified period.

        Parameters:
        asset (Asset):  The Asset class for the asset.
        period (int): The number of days over which to calculate the ATR.

        Returns:
        float: The ATR value if available, or None if necessary data is missing.
        """

        historical_prices = self.get_historical_prices(asset, length=period + 1)
        if historical_prices is None or 'high' not in historical_prices.df.columns or 'low' not in historical_prices.df.columns or 'close' not in historical_prices.df.columns:
            return None

        df = historical_prices.df
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.Series(np.maximum.reduce([high_low, high_close, low_close]))

        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr
    
    def calculate_rsi(self, asset, period=14):
        """
        Calculates the Relative Strength Index (RSI) for a given asset over a specified period.

        Parameters:
        asset (Asset):  The Asset class for the asset.
        period (int): The number of days over which to calculate the RSI.

        Returns:
        float: The RSI value if available, or None if there are no historical close prices.
        """

        historical_prices = self.get_historical_prices(asset, length=period+1)
        if historical_prices is None or 'close' not in historical_prices.df.columns:
            self.log_message(f"No historical close prices available for {asset.symbol}.")
            return None
        
        prices_df = historical_prices.df
        close = prices_df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_bollinger_bands(self, asset, period=20, num_std=2):
        """
        Calculates Bollinger Bands for a given asset over a specified period.

        Parameters:
        asset (Asset):  The Asset class for the asset.
        period (int): The number of days for calculating the simple moving average (SMA) at the center of the bands.
        num_std (int): The number of standard deviations to determine the upper and lower bands.

        Returns:
        tuple: A tuple containing the upper band, SMA (middle band), and lower band values. Returns None if necessary data is missing.
        """

        historical_prices = self.get_historical_prices(asset, length=period)
        if historical_prices is None or 'close' not in historical_prices.df.columns:
            self.log_message(f"No historical close prices available for {asset.symbol}.")
            return None

        prices_df = historical_prices.df
        sma = prices_df['close'].rolling(window=period).mean()
        std_dev = prices_df['close'].rolling(window=period).std()

        upper_band = sma + (std_dev * num_std)
        lower_band = sma - (std_dev * num_std)

        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
    def calculate_order_blocks(self, ohlc: pd.DataFrame, lookback_period=3) -> pd.DataFrame:
        """
        Identifies Order Blocks in a series of OHLC (Open, High, Low, Close) price data based on significant volume changes.

        Parameters:
        ohlc (pd.DataFrame): DataFrame containing OHLC price data.
        lookback_period (int): The number of candles to look back to identify a significant change for the definition of an Order Block.

        Returns:
        pd.DataFrame: A DataFrame with 'OrderBlockType' and 'OrderBlockLevel' columns, indicating the type and level of detected Order Blocks.
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
        
    def klinger_signal_crossover(self, asset, crossover_type='bullish', short_window=34, long_window=55, signal_window=13):
        """
        Identifies crossovers between the Klinger Volume Oscillator and its signal line,
        which can indicate potential buy or sell opportunities based on the specified crossover type.

        Parameters:
        asset (Asset): The Asset class for the asset.
        crossover_type (str): 'bullish' for bullish crossover, 'bearish' for bearish crossover.
        short_window (int): The short period for the KVO calculation.
        long_window (int): The long period for the KVO calculation.
        signal_window (int): The period for the signal line (EMA of the KVO).

        Returns:
        bool: True if a crossover is detected matching the specified type, False otherwise.
        """
        # Get historical data required to compute the KVO
        historical_prices = self.get_historical_prices(asset, length=max(short_window, long_window, signal_window) + long_window)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns and 'volume' in prices_df.columns:
            # Calculate the KVO and its signal line
            short_window = short_window*self.multiplier
            long_window = long_window*self.multiplier
            signal_window = signal_window*self.multiplier
            kvo_df = ta.kvo(prices_df['high'], prices_df['low'], prices_df['close'], prices_df['volume'], fast=short_window, slow=long_window, signal=signal_window)

            # Extract the latest and previous KVO and signal values for comparison
            latest_kvo = kvo_df[f"KVO_{short_window}_{long_window}_{signal_window}"].iloc[-1]
            previous_kvo = kvo_df[f"KVO_{short_window}_{long_window}_{signal_window}"].iloc[-2]
            latest_signal = kvo_df[f"KVOs_{short_window}_{long_window}_{signal_window}"].iloc[-1]
            previous_signal = kvo_df[f"KVOs_{short_window}_{long_window}_{signal_window}"].iloc[-2]

            # Detect bullish crossover
            if crossover_type == 'bullish':
                if previous_kvo < previous_signal and latest_kvo > latest_signal:
                    self.log_message(f"{asset.symbol}: KVO crossed above signal line (bullish).")
                    return True
            # Detect bearish crossover
            elif crossover_type == 'bearish':
                if previous_kvo > previous_signal and latest_kvo < latest_signal:
                    self.log_message(f"{asset.symbol}: KVO crossed below signal line (bearish).")
                    return True
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns or 'volume' not in prices_df.columns:
                self.log_message(f"Missing 'close' or 'volume' column for {asset.symbol}.")

        return False
    
    def klinger_threshold_check(self, asset, line_type='klinger', threshold=0, above=True, short_window=34, long_window=55, signal_window=13):
        """
        Checks if the specified Klinger line(s) is/are above or below a given threshold.

        Parameters:
        asset (Asset): The Asset class for the asset.
        line_type (str): Specifies which line to check, 'klinger' for the Klinger line, 'signal' for the signal line, or 'both' for both lines.
        threshold (float): The threshold value to compare against.
        above (bool): If True, checks if the line(s) is/are above the threshold; if False, checks if below.
        short_window (int): The short period for the KVO calculation.
        long_window (int): The long period for the KVO calculation.
        signal_window (int): The period for the signal line (EMA of the KVO).

        Returns:
        bool: True if the specified line(s) is/are above (or below, based on 'above' param) the threshold, False otherwise.
        """
        # Get historical data required to compute the KVO
        length = max(short_window, long_window, signal_window) + long_window
        historical_prices = self.get_historical_prices(asset, length=length)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns and 'volume' in prices_df.columns:
            # Calculate the KVO and its signal line
            short_window = short_window*self.multiplier
            long_window = long_window*self.multiplier
            signal_window = signal_window*self.multiplier
            kvo_df = ta.kvo(prices_df['high'], prices_df['low'], prices_df['close'], prices_df['volume'], fast=short_window, slow=long_window, signal=signal_window)
            latest_kvo = kvo_df[f"KVO_{short_window}_{long_window}_{signal_window}"].iloc[-1]
            latest_signal = kvo_df[f"KVOs_{short_window}_{long_window}_{signal_window}"].iloc[-1]

            # Determine if the latest value of the specified line is above or below the threshold
            if line_type == 'both':
                condition_kvo = (latest_kvo > threshold) if above else (latest_kvo < threshold)
                condition_signal = (latest_signal > threshold) if above else (latest_signal < threshold)
                return condition_kvo and condition_signal
            elif line_type == 'klinger':
                return (latest_kvo > threshold) if above else (latest_kvo < threshold)
            elif line_type == 'signal':
                return (latest_signal > threshold) if above else (latest_signal < threshold)
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns or 'volume' not in prices_df.columns:
                self.log_message(f"Missing 'close' or 'volume' column for {asset.symbol}.")

        return False
    
    def klinger_vs_signal(self, asset, above=True, short_window=34, long_window=55, signal_window=13):
        """
        Checks if the Klinger oscillator line is above or below its signal line.

        Parameters:
        asset (Asset): The Asset class for the asset.
        above (bool): If True, checks if the Klinger line is above the signal line; if False, checks if it is below.
        short_window (int): The short period for the KVO calculation.
        long_window (int): The long period for the KVO calculation.
        signal_window (int): The period for the signal line (EMA of the KVO).

        Returns:
        bool: True if the Klinger line is above (or below, based on 'above' param) the signal line, False otherwise.
        """
        # Get historical data required to compute the KVO
        length = (max(short_window, long_window, signal_window) + long_window)
        historical_prices = self.get_historical_prices(asset, length=length)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns and 'volume' in prices_df.columns:
            # Calculate the KVO and its signal line
            short_window = short_window*self.multiplier
            long_window = long_window*self.multiplier
            signal_window = signal_window*self.multiplier

            kvo_df = ta.kvo(prices_df['high'], prices_df['low'], prices_df['close'], prices_df['volume'], fast=short_window, slow=long_window, signal=signal_window)
            latest_kvo = kvo_df[f"KVO_{short_window}_{long_window}_{signal_window}"].iloc[-1]
            latest_signal = kvo_df[f"KVOs_{short_window}_{long_window}_{signal_window}"].iloc[-1]

            # Determine if the Klinger line is above or below the signal line
            return (latest_kvo > latest_signal) if above else (latest_kvo < latest_signal)
        else:
            if prices_df is None:
                self.log_message(f"No historical prices data found for {asset.symbol}.")
            elif 'close' not in prices_df.columns or 'volume' not in prices_df.columns:
                self.log_message(f"Missing 'close' or 'volume' column for {asset.symbol}.")

        return False


    def avoid_whipsaw_exit(self, asset, indicator_length=20, indicator_type='SMA', atr_multiplier=1):
        """
        Evaluates if the current price of the asset is within a band of +/- ATR around the indicator value at the time of entry,
        to avoid exiting during whipsaws in trend following strategies.

        Parameters:
        asset (Asset): The asset being evaluated.
        indicator_length (int): The length of the indicator (e.g., moving average).
        indicator_type (str): The type of indicator, either 'SMA' or 'EMA'.
        atr_multiplier (float): The multiplier for the ATR to set the band limits.

        Returns:
        bool: False if the current price is within the band, indicating no exit should occur. True otherwise.
        """
        position = self.get_position(asset)
        if position is None or not position.orders:
            self.log_message(f"No position or orders found for {asset.symbol}.")
            return False

        # Find the last buy order that was filled
        entry_indicator_value = None
        entry_price = None
        entry_datetime = None
        for order in reversed(position.orders):
            if order.side == 'buy' and order.is_filled():
                entry_price = order.get_fill_price()
                entry_datetime = order.get_filled_datetime()
                break

        if entry_price is None or entry_datetime is None:
            self.log_message(f"No filled buy orders found for {asset.symbol}.")
            return False

        # Get historical prices up to the entry time
        historical_prices = self.get_historical_prices(asset, length=indicator_length, end_date=entry_datetime)
        prices_df = historical_prices.df if historical_prices else None

        if prices_df is not None and 'close' in prices_df.columns:
            if indicator_type == 'SMA':
                indicator_values = prices_df['close'].rolling(window=indicator_length).mean()
            elif indicator_type == 'EMA':
                indicator_values = prices_df['close'].ewm(span=indicator_length, adjust=False).mean()
            else:
                self.log_message(f"Unsupported indicator type: {indicator_type}")
                return False

            entry_indicator_value = indicator_values.iloc[-1]

        if entry_indicator_value is None:
            self.log_message(f"Unable to calculate indicator value at entry for {asset.symbol}.")
            return False

        current_date = self.get_datetime().date()
        cache_key = self.generate_cache_key("avoid_whipsaw_exit", asset.symbol, current_date, entry_indicator_value, atr_multiplier)

        if cache_key in self.signal_history:
            return self.signal_history[cache_key]

        atr = self.calculate_atr(asset)
        if atr is None:
            self.log_message(f"Unable to calculate ATR for {asset.symbol}.")
            return False

        upper_band = entry_indicator_value + atr_multiplier * atr
        lower_band = entry_indicator_value - atr_multiplier * atr
        current_price = self.get_last_price(asset)

        if lower_band <= current_price <= upper_band:
            self.log_message(f"{asset.symbol}: Current price {current_price} is within the band ({lower_band}, {upper_band}). No exit.")
            signal_triggered = False
        else:
            self.log_message(f"{asset.symbol}: Current price {current_price} is outside the band ({lower_band}, {upper_band}). Exit signal triggered.")
            signal_triggered = True

        self.signal_history[cache_key] = signal_triggered
        return signal_triggered
    

    ### Helper methods ###
    def get_multiplier(self):
        return self._sleeptime_to(timestep=self.get_timestep(), sleeptime=self.get_sleeptime())
    
    def _sleeptime_to(self, timestep=None, sleeptime=None):
        """Convert the sleeptime according to the timestep provided ('minute' or 'day'), ensuring days are returned as integers."""
        val_err_msg = ("You can set the sleep time as an integer which will be interpreted as minutes. "
                    "For example, sleeptime = 50 would be 50 minutes. Conversely, you can enter the time as a string "
                    "with the duration numbers first, followed by the time units: 'M' for minutes, 'S' for seconds, "
                    "'H' for hours, 'D' for days, e.g., '300S' is 300 seconds.")

        # Default conversion is to minutes
        if isinstance(sleeptime, int):
            minutes = sleeptime
        elif isinstance(sleeptime, str):
            unit = sleeptime[-1].lower()
            time_raw = int(sleeptime[:-1])

            if unit == "s":
                minutes = time_raw // 60
            elif unit == "m":
                minutes = time_raw
            elif unit == "h":
                minutes = time_raw * 60
            elif unit == "d":
                minutes = time_raw * 1440  # 24 * 60
            else:
                raise ValueError(val_err_msg)
        else:
            raise ValueError(val_err_msg)

        # Convert minutes to days if required
        if timestep.lower() == "day":
            return math.floor(minutes / 1440)
        elif timestep.lower() == "minute":
            return minutes
        else:
            raise ValueError("Invalid timestep. Please use 'minute' or 'day'.")