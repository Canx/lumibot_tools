class Sizing:

    MAX_CRYPTO_DECIMALS = 8

    # TODO: Specify buy and sell sizing strategy in params!
    def __init__(self, strategy, signals, min_cash, assets=None):
        self.cash = strategy.cash
        self.assets = assets
        self.min_cash = min_cash
        self.max_crypto_decimals = Sizing.MAX_CRYPTO_DECIMALS
        self.get_historical_prices = strategy.get_historical_prices
        self.log_message = strategy.log_message
        self.signals = signals
        self.max_percentage_per_trade=1.0

    def shares_to_buy(self, asset, price_per_share, method='basic', **kwargs):
        if price_per_share <= 0:
                return 0.0
    
        if self.cash < self.min_cash:
                return 0.0
        
        if method == 'basic':
            return self.shares_to_buy_basic(asset, price_per_share, **kwargs)
        elif method == 'var':
            return self.shares_to_buy_var(asset, price_per_share, **kwargs)
        elif method == 'trend':
            return self.shares_to_buy_trend(asset, price_per_share, **kwargs)
        else:
            raise ValueError("Unsupported method type")
    
    # TODO: Create different sell strategies
    def shares_to_sell(self, position):
         if position.quantity > 0:
             # Sell all positions
             return position.quantity

    def shares_to_buy_basic(self, asset, price_per_share):
        
            # Calcula el cash disponible para esta operación según el máximo porcentaje permitido
            available_cash = self.cash * self.max_percentage_per_trade 

            # Divide el cash disponible entre el número de activos elegibles
            available_cash_per_asset = available_cash / len(self.assets)

            if asset.asset_type == "crypto":
                # Calcula y redondea la cantidad de criptomoneda a comprar según el máximo de decimales permitido
                shares_to_buy = available_cash_per_asset / price_per_share
                shares_to_buy = round(shares_to_buy, Sizing.MAX_CRYPTO_DECIMALS)
            elif asset.asset_type == "stock":
                # Para acciones, redondea al número entero más cercano
                shares_to_buy = int(available_cash_per_asset / price_per_share)
            else:
                raise ValueError("Unsupported asset type")

            return shares_to_buy
    
    def shares_to_buy_trend(self, asset, price_per_share, trend_up=None):
        
        trend_function = trend_up
        if not trend_up:
            trend_function = self.trend_up_ema

        if price_per_share <= 0:
            return 0.0
        
        if self.cash < self.min_cash:
            return 0.0
        
        available_cash = self.cash * self.max_percentage_per_trade * trend_function(asset)
        
        if asset.asset_type == "crypto":
            # Calcula y redondea la cantidad de criptomoneda a comprar según el máximo de decimales permitido
            shares_to_buy = available_cash / price_per_share
            shares_to_buy = round(shares_to_buy, self.max_crypto_decimals)
        elif asset.asset_type == "stock":
            # Para acciones, redondea al número entero más cercano
            shares_to_buy = int(available_cash / price_per_share)
        else:
            raise ValueError("Unsupported asset type")

        return shares_to_buy
    
    # Trend up strentgh (from 0 to 1)
    def trend_up_ema(self, asset):
        trend_signals = [
            self.signals.price_above_below_EMA(asset, length=20, position='above'),
            self.signals.short_over_long_ma(asset, short_length=20, long_length=50, ma_type='EMA'),
            self.signals.short_over_long_ma(asset, short_length=50, long_length=100, ma_type='EMA'),
            self.signals.short_over_long_ma(asset, short_length=100, long_length=200, ma_type='EMA')
        ]

        # Calcular la fuerza de la tendencia como la suma de las señales de tendencia
        trend_strength = sum(trend_signals)

        # Normalizar la fuerza de la tendencia para que esté en el rango de 0 a 1
        normalized_trend_strength = trend_strength / len(trend_signals)

        return normalized_trend_strength