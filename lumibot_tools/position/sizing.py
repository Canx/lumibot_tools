class Sizing:

    MAX_CRYPTO_DECIMALS = 8

    def __init__(self, strategy, min_cash, trend_up=None, trend_down=None, assets=None):
        self.cash = strategy.cash
        self.assets = assets
        self.min_cash = min_cash
        self.max_crypto_decimals = Sizing.MAX_CRYPTO_DECIMALS
        self.get_historical_prices = strategy.get_historical_prices
        self.log_message = strategy.log_message
        self.trend_up = trend_up
        self.trend_down = trend_down

    def shares_to_buy(self, asset, price_per_share, method='basic', **kwargs):
        if method == 'basic':
            return self.shares_to_buy_basic(asset, price_per_share, self.cash, **kwargs)
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

    def shares_to_buy_basic(self, asset, price_per_share, cash, max_percentage_per_trade=1.0):
            
            if price_per_share <= 0:
                return 0.0
            
            if cash < self.min_cash:
                return 0.0

            # Calcula el cash disponible para esta operación según el máximo porcentaje permitido
            available_cash = cash * max_percentage_per_trade 

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
    
    def shares_to_buy_trend(self, asset, price_per_share, trend_up=None, max_percentage_per_trade=1.0):
        if price_per_share <= 0:
            return 0.0
        
        if self.cash < self.min_cash:
            return 0.0
        
        available_cash = self.cash * max_percentage_per_trade * trend_up(asset)
        
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