class Sizing:

    MAX_CRYPTO_DECIMALS = 8
    
    def shares_to_buy_basic(asset, price_per_share, cash, min_cash, eligible_assets, max_percentage_per_trade=1.0):
            if price_per_share <= 0:
                return 0.0
            
            if cash < min_cash:
                return 0.0

            # Calcula el cash disponible para esta operación según el máximo porcentaje permitido
            available_cash = cash * max_percentage_per_trade 

            # Divide el cash disponible entre el número de activos elegibles
            available_cash_per_asset = available_cash / eligible_assets

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