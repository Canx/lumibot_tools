from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset
from lumibot.backtesting import PolygonDataBacktesting
from lumibot_tools.messaging import MessagingStrategy, TelegramBotHandler
import datetime
import pandas as pd
from math import floor


class TurtleStrategy(MessagingStrategy):

    SYSTEM_1 = 1
    SYSTEM_2 = 2
    # Determina el número máximo de unidades permitidas
    MAX_UNITS = 12  # Ejemplo: máximo de 4 unidades, a 2% por unidad: 24%

    def initialize(self, parameters=None):
        self.sleeptime = "1D"
        self.minutes_before_closing = 5
        #self.assets = ["SPY", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
        self.assets = self.get_assets()
        self.max_assets = 20 # Maximum number of assets to trade
        self.max_initial_position = 0.05 # Maximum percentage of cash in initial position per asset
        self.position_metadata = {}  # Clave: ID de posición o símbolo, Valor: diccionario de metadatos
        self.breakout_period = 20
        self.breakout_long_period = 55
        self.exit_period = 10  # Periodo para el cálculo de la señal de salida
        self.atr_period = 14  # Periodo típico para calcular el ATR
        self.risk_per_trade = 0.005  # Riesgo del 0.5% por operación
        self.risk_multiplier = 2 # Definir el múltiplo del ATR para calcular el stop loss (p. ej., 2xATR)
        self.dollars_per_point = 1 # Para acciones es 1
        self.last_breakout_results = {}  # Ejemplo: {'AAPL_System1': {'was_winner': True, 'direction': 'long'}}
        self.position_metadata = {}
        self.current_system_is_1 = True

    def get_assets(self):
        if self.is_backtesting:
            return ["SPY"]
        else:
            # TODO: Añadir assets actuales del broker 
            from finvizfinance.screener.overview import Overview

            foverview = Overview()
            filters_dict = {
                'Current Volume': 'Over 1M',
                'Beta': 'Over 1',
                'Performance': 'Month +10%',
                'Performance 2': 'Year +30%',
                'RSI (14)': 'Not Overbought (<60)',
                '200-Day Simple Moving Average': 'Price above SMA200',
                '50-Day Simple Moving Average': 'Price above SMA50',
                'Gap': 'Up'
            }
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view()

            # Verificar si el DataFrame está vacío
            if not df.empty:
                # Si el DataFrame no está vacío, extraer la lista de tickers
                symbol_list = df['Ticker'].tolist()
                self.log_message("Símboloes screener:", symbol_list)
            else:
                # Si el DataFrame está vacío, manejar adecuadamente
                symbol_list = []
                print("No se encontraron símbolos en el screener.")
        
            # Añadimos los símbolos de las posiciones actuales
            symbol_list = self.get_position_symbols() + symbol_list

            # Quitamos duplicados manteniendo el orden
            symbol_list = list(dict.fromkeys(symbol_list))
        
            return symbol_list

    def get_position_symbols(self):
        # Obtenemos la lista de posiciones
        positions = self.get_positions()
        
        # Usamos una comprensión de lista para extraer los símbolos
        symbols = [position.symbol for position in positions]
        
        return symbols

    def on_trading_iteration(self):
        historical_data = self.get_and_check_historical_data()

        self.check_for_stoploss(historical_data, TurtleStrategy.SYSTEM_2)
        self.check_for_exits(historical_data, TurtleStrategy.SYSTEM_2)
        self.check_for_entries(historical_data, TurtleStrategy.SYSTEM_2)
        
        #self.check_for_entries(historical_data, TurtleStrategy.SYSTEM_1)
        #self.check_for_exits(historical_data, TurtleStrategy.SYSTEM_1)
        #if self.current_system_is_1:
            
        #else:
            

        #self.current_system_is_1 = not(self.current_system_is_1)
        
    def get_and_check_historical_data(self):
        length = max(self.breakout_period, self.exit_period, self.atr_period) + self.atr_period + 1
        historical_data = {}

        for asset in self.assets:
            bars = self.get_historical_prices(asset=asset, length=length, timestep="day")
            
            # Verifica si se recuperaron datos para el activo
            if bars is None or bars.df.empty:
                self.log_message(f"No se encontraron datos históricos para {asset}.")
                continue
            
            # Agrega los datos recuperados al diccionario historical_data
            historical_data[asset] = bars

        return historical_data

    def was_last_breakout_winner(self, asset, system):
        key = f"{asset}_{system}"
        if key in self.last_breakout_results:
            was_breakout = self.last_breakout_results[key]
            # Lo borramos para que solo se evite 1 vez
            del self.last_breakout_results[key]
            return was_breakout
        else:
            # Si no hay información previa del breakout, podríamos asumir que no fue ganador
            # para permitir una operación basada en la nueva señal de breakout.
            return False

    def calculate_extra_position_size(self, current_price, last_entry_price, atr, system_units):
        HALF_N = atr / 2  # 1/2 N para añadir una unidad.
        price_movement = current_price - last_entry_price

        # Esto calcula cuántas medias N caben en el movimiento de precio
        potential_units_to_add = int(price_movement / HALF_N)  

        # Asegúrate de no superar el máximo de unidades permitidas
        actual_units_to_add = min(potential_units_to_add, self.MAX_UNITS - system_units)

        if actual_units_to_add > 0:
            return actual_units_to_add
        else:
            return 0

    
    def calculate_initial_position_size(self, asset, atr):
        # Obtener el último precio del activo
        current_price = self.get_last_price(asset)

        if current_price is None:
            self.send_message(f"No se pudo obtener el último precio para {asset}.")
            return 0

        # Obtener el capital total de la cuenta
        total_capital = self.get_cash()

        # Asegurar que el capital total no sea negativo
        if total_capital <= 0:
            self.log_message(f"Capital insuficiente: {total_capital}")
            return 0

        self.log_message(f"Capital: {total_capital}")
        # Calcular el monto del riesgo en dinero
        risk_amount = total_capital * self.risk_per_trade
        self.log_message(f"Cantidad a arriesgar máxima: {risk_amount}")

        # Calcular la volatilidad en dólares del activo
        dollar_volatility = atr * self.dollars_per_point

        self.log_message(f"Volatilidad de dolar: {dollar_volatility}")
        # Calcular el tamaño de la posición basado en la volatilidad del activo y el riesgo permitido
        position_size = risk_amount / dollar_volatility

        # Ajuste para no superar el 50% del efectivo disponible
        max_cash_to_use = total_capital * self.max_initial_position  # Usar solo hasta el 5% del efectivo disponible
        cash_needed_for_position = current_price * position_size  # Calcula el efectivo necesario para la posición calculada

        # Si el efectivo necesario supera el máximo permitido, ajustar la posición
        if cash_needed_for_position > max_cash_to_use:
            position_size = max_cash_to_use / current_price
            self.log_message("Posición calculada supera el 50% del cash! Ajustando...")

        # Redondear hacia abajo para obtener solo unidades enteras
        units_to_buy = floor(position_size)
        self.log_message(f"Unidades a comprar: {units_to_buy}")

        return units_to_buy


    def calculate_atr(self, df):
        # Calculamos el True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calcula el promedio simple de True Range para los primeros 20 días como punto de partida
        initial_n = true_range.rolling(window=20).mean().iloc[19]
        
        # Calcula el EMA de True Range para el resto de los días
        n = true_range.ewm(alpha=1/20, adjust=False).mean()
        
        # Reemplaza los primeros 19 valores de N con el promedio inicial para alinear con la estrategia clásica de las Tortugas
        n.iloc[:20] = initial_n
        
        return n.iloc[-1]


    def get_breakout_levels(self, df, system):
        # Determina el período basado en el sistema
        period = 10 if system == TurtleStrategy.SYSTEM_1 else 20
        
        breakout_long = df['high'].rolling(window=period).max().iloc[-2]  # -2 para usar el penúltimo valor, asegurando que el breakout esté confirmado al final del día anterior
        breakout_short = df['low'].rolling(window=period).min().iloc[-2]  # Lo mismo para el breakout corto
    
        return breakout_long, breakout_short

    def calculate_stop_loss(self, current_price, atr, direction):
        stop_loss = None
        if direction == "long":
            stop_loss = current_price - (atr * self.risk_multiplier)
        elif direction == "short":
            stop_loss = current_price + (atr * self.risk_multiplier)
        else:
            self.log_message("Error calculating stop_loss")

        return stop_loss

    # Comprobar cada posición y ver si ha llegado a su stop-loss, en cuyo caso lanzar orden de mercado.
    def check_for_stoploss(self, historical_data, system):
        pass

    def check_for_entries(self, historical_data, system):
        for asset, data in historical_data.items():
            df = data.df

            self.evaluate_entry_system(asset, df, system)

    def evaluate_entry_system(self, asset, df, system):
        # Verifica si el último breakout fue un ganador
        if self.was_last_breakout_winner(asset, system) and system == TurtleStrategy.SYSTEM_1:
            self.log_message(f"Ignorando señal de entrada para {asset} en el sistema {system} debido a un breakout ganador anterior.")
            return  # Ignora la entrada si el último breakout fue un ganador
        
        current_price = self.get_last_price(asset)
        atr = self.calculate_atr(df)
        
        position = self.get_position(asset)
        key = f"{asset}_{system}"

        # Inicializa variables a usar basadas en position_metadata
        system_quantity = 0
        system_units = 0
        system_last_price = None

        # Si existe una entrada para este asset y sistema en position_metadata, la utiliza
        if key in self.position_metadata:
            metadata = self.position_metadata[key]
            system_quantity = metadata.get('units', 0)
            system_units = metadata.get('units', 0)  # Asumiendo que 'units' representa tanto la cantidad como las "unidades" para la lógica de tu estrategia
            system_last_price = metadata.get('last_price', None)  # Asegúrate de actualizar este valor en tu lógica de trading
        
        system_prefix = "s1" if system == TurtleStrategy.SYSTEM_1 else "s2"
        system_quantity = getattr(position, f"{system_prefix}_quantity", 0)
        system_units = getattr(position, f"{system_prefix}_units", 0)
        system_last_price = getattr(position, f"{system_prefix}_last_unit_price", None)
        
        direction = None
        stop_loss_price = None
        unit_quantity = 0
        
        # Diferenciamos si ya estamos en una posición o no!
        if system_quantity > 0:
            self.log_message(f"evaluate extra position signals {key}")
            unit_quantity = self.calculate_extra_position_size(current_price, system_last_price, atr, system_units)
            
            if unit_quantity > 0:
                direction = "long" if system_quantity > 0 else "short"
                stop_loss_price = self.calculate_stop_loss(current_price, atr, direction)
                self.log_message(f"Añadiendo a posición {asset} {direction} {unit_quantity}")

        else:
            self.log_message(f"evaluate entry signals {key}")
            # Determina si se debería abrir una posición
            direction = self.check_direction_breakout(current_price, df, system)
            
            # Si hay dirección determinada, procede a abrir a la posición
            if direction:
                # Determina el número de unidades a abrir basado en el ATR y el tamaño de posición calculado
                unit_quantity = self.calculate_initial_position_size(asset, atr)
                stop_loss_price = self.calculate_stop_loss(current_price, atr, direction)
                
            # Llama directamente a open_position con los parámetros necesarios
        self.open_position(asset, 
                           direction, 
                           unit_quantity, 
                           current_price, 
                           stop_loss_price, 
                           None,  # take_profit_price, si decides usarlo
                           system, 
                           True)  # Supongo que esto indica si es una operación de breakout o adición de unidades

    def check_direction_breakout(self, current_price, df, system):
        direction = None
        breakout_long, breakout_short = self.get_breakout_levels(df, system)
        if current_price >= breakout_long:
            direction = "long"
        
        # BUG! Los shorts no van bien!!
        #elif current_price <= breakout_short:
        #    direction = "short"

        return direction
    
    def should_add_to_position(self, asset, atr, system_quantity, last_unit_price, system_units):
        # Asume que solo se añade 1 unidad por vez, siguiendo la estrategia original de las Tortugas.
        MAX_UNITS = 4  # Ejemplo: máximo de 4 unidades permitidas.
        HALF_N_MOVEMENT_REQUIRED = atr / 2  # 1/2 N para añadir una unidad.
        
        current_price = self.get_last_price(asset)
        
        # Calcula la diferencia de precio desde la última unidad añadida.
        price_difference = current_price - last_unit_price if system_quantity > 0 else last_unit_price - current_price
        
        # Verifica si el movimiento del precio es suficiente para añadir una nueva unidad y si no se ha alcanzado el máximo de unidades.
        if price_difference >= HALF_N_MOVEMENT_REQUIRED and system_units < MAX_UNITS:
            # Aquí, asumimos que se añade una unidad a la vez.
            return True  # Añade una unidad.
        else:
            return False  # No añade ninguna unidad si no se cumplen las condiciones.

    def check_for_exits(self, historical_data, system):
        for position in self.get_positions():
            symbol = position.symbol
            if symbol not in historical_data or historical_data[symbol] is None:
                self.log_message(f"Saltando {symbol} debido a la falta de datos históricos.")
                continue

            df = historical_data.get(symbol).df
            self.evaluate_exit_signals(position, df, system)

    def evaluate_exit_signals(self, position, df, system):
        
        key = f"{position.symbol}_{system}"
        
        self.log_message(f"evaluate_exit_signals {key}")
        # Inicializa la cantidad a 0 por defecto
        quantity = 0
    
        # Verifica si existe la clave en position_metadata para obtener la cantidad de unidades
        if key in self.position_metadata:
            quantity = self.position_metadata[key].get('quantity', 0)
        
        self.log_message(f"{quantity} acciones en {key}")
        # Continúa solo si hay una posición activa (cantidad no es 0)
        if quantity != 0:
            self.log_message(f"{quantity} acciones en {key}")
            if self.exit_signal(df, system, quantity):
                self.close_position(position, system)


    def exit_signal(self, df, system_used, quantity):
        period = 10 if system_used == TurtleStrategy.SYSTEM_1 else 20
        exit_price = df['low'].rolling(window=period).min().iloc[-2] if quantity > 0 \
                    else df['high'].rolling(window=period).max().iloc[-1]
        current_price = df['close'].iloc[-1]

        return current_price < exit_price if quantity > 0 else current_price > exit_price

    def open_position(self, asset_symbol, direction, quantity, entry_price, stop_loss_price, take_profit_price, system_used, is_breakout):
        if quantity == 0:
            #self.log_message(f"No se puede abrir una posición para {asset_symbol} debido a que la cantidad calculada es 0.")
            return

        if direction is None:
            return
        
        # Traduce la dirección de 'long'/'short' a 'buy'/'sell'
        order_side = "buy" if direction == "long" else "sell"
        
        # Comprobar si tenemos suficiente cash
        cash_needed_for_position = entry_price * quantity  # Calcula el efectivo necesario para la posición calculada

        # Si el efectivo necesario supera el máximo permitido, ajustar la posición
        if cash_needed_for_position > self.get_cash():
            self.log_message(f"No es posible abrir posición, no hay suficiente cash: {self.get_cash()}")
            return

        self.log_message(f"Entrando en {direction} con {quantity} de {asset_symbol}")
        
        # Crear la orden
        order = self.create_order(
            asset=Asset(asset_symbol),
            quantity=abs(quantity),
            side=order_side,
            limit_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            custom_params={'system_used': system_used,
                           'is_breakout': is_breakout}
        )
        
        self.submit_order(order)
        #self.wait_for_order_execution(order)
    
    def close_position(self, position, system_used, current_price=None):
        key = f"{position.symbol}_{system_used}"
        metadata = self.position_metadata[key]
        quantity = metadata["quantity"]

        if quantity != 0:
            # Determinar la acción de la orden basada en si la posición es long o short
            order_action = "buy" if quantity < 0 else "sell"
            # Usar el valor absoluto de quantity_to_close para asegurar que la cantidad es positiva
            close_order = self.create_order(position.symbol,
                                           abs(quantity),
                                           order_action,
                                           custom_params={'system_used': system_used,
                                                          'is_breakout': False})
            
            self.log_message(f"Cerrando posición para {key} con {abs(quantity)} unidades.")
            self.submit_order(close_order)
            #self.wait_for_order_execution(close_order)

    def calculate_position_profit_loss(self, key):
        metadata = self.position_metadata[key]
        balance = metadata['sales_revenue'] - metadata['cost']
        return balance

    # Aquí deberíamos calcular y guardar información de la posición
    # Cantidad ganada, winner/losser,
    def on_filled_order(self, position, order, price, quantity, multiplier):

        system_used = order.custom_params['system_used']
        key = f"{position.symbol}_{system_used}"

        if key not in self.position_metadata:
            self.position_metadata[key] = {'cost': 0, 'quantity': 0, 'sales_revenue': 0}
        
        detail_text = f"{quantity}x{price} of {position.symbol}. System: {system_used}"
        if order.side == 'buy':
            self.position_metadata[key]['cost'] += price * quantity
            self.position_metadata[key]['quantity'] += quantity
            self.position_metadata[key]['last_price'] = price
            color = "blue" if system_used == 1 else "white"
            self.add_marker("Buy", symbol="triangle-up", value=price, color=color, detail_text=detail_text)
        elif order.side == 'sell':
            self.position_metadata[key]['sales_revenue'] += price * quantity
            self.position_metadata[key]['quantity'] -= quantity
            if self.position_metadata[key]['quantity'] == 0:
                # Determina si la subposición fue ganadora y realiza acciones adecuadas
                balance = self.calculate_position_profit_loss(key)
                
                color = "green" if balance > 0 else "red"
                self.add_marker("Closed", symbol="circle", value=price, color=color, detail_text=detail_text + f" Balance: {balance}")
                # Actualiza last_breakout_results con el resultado del último trade
                self.last_breakout_results[key] = balance > 0
        
                # Ya que la posición está cerrada, podrías decidir limpiar los datos de posición_metadata para este key
                del self.position_metadata[key]
        
        #self.log_message(f"Position metadata for {key}:{self.position_metadata[key]}")
if __name__ == "__main__":
    is_live = False

    if is_live:
        from lumibot.brokers import Alpaca
        from config import ALPACA_CONFIG, TELEGRAM_CONFIG
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = TurtleStrategy(broker)

        # Set telegram bot and attach to strategy
        bot = TelegramBotHandler(TELEGRAM_CONFIG)
        strategy.set_messaging_bot(bot)

        # Set trader
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        from config import POLYGON_CONFIG
        backtesting_start = datetime.datetime(2023, 6, 1)
        backtesting_end = datetime.datetime(2023, 12, 1)
        trading_fee = TradingFee(percent_fee=0.001)

        trader = Trader(backtest=True)
        trading_fee = TradingFee(percent_fee=0.001)
        TurtleStrategy.backtest(
            PolygonDataBacktesting,
            backtesting_start,
            backtesting_end,
            polygon_api_key=POLYGON_CONFIG["KEY"],
            polygon_has_paid_subscription=POLYGON_CONFIG["PAID"],
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            benchmark_asset="SPY",
            risk_free_rate=0.532)
