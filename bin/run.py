#!/usr/bin/env python
import sys, os
import argparse
import importlib
import inspect
from datetime import datetime, timedelta
from lumibot.traders import Trader
from lumibot.entities import TradingFee, Asset

sys.path.insert(0, os.getcwd())
import config

def parse_arguments():
    """Parse command line arguments for running the strategy."""
    parser = argparse.ArgumentParser(description="Run a trading strategy.")
    parser.add_argument("strategy_file", help="The filename of the strategy to run")
    parser.add_argument("--live", action="store_true", help="Run the strategy in live mode")
    parser.add_argument("--broker", default='Kraken', choices=['IB', 'Kraken','Alpaca'], help="Broker to use for live trading (Interactive Brokers or Kraken)")
    parser.add_argument("--start", help="Backtesting start date in YYYY-MM-DD format")
    parser.add_argument("--end", help="Backtesting end date in YYYY-MM-DD format")
    return parser.parse_args()

def find_strategy_class(module):
    """Find and return the first strategy class in the given module."""
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            return obj
    return None

def configure_broker():
    """Auto-detect and configure the broker for live trading based on available credentials."""
    try:
        from credentials import INTERACTIVE_BROKERS_CONFIG
        from lumibot.brokers import InteractiveBrokers
        print("Using Interactive Brokers for live trading.")
        return InteractiveBrokers(INTERACTIVE_BROKERS_CONFIG)
    except ImportError:
        pass

    try:
        from credentials import KRAKEN_CONFIG
        from lumibot.brokers import Ccxt
        print("Using Kraken for live trading.")
        return Ccxt(KRAKEN_CONFIG)
    except ImportError:
        pass

    try:
        from credentials import ALPACA_CONFIG
        from lumibot.brokers import Alpaca
        print("Using Alpaca for live trading.")
        return Alpaca(ALPACA_CONFIG)
    except ImportError:
        pass

    raise ImportError("No broker configuration found in credentials.py.")

def configure_quote(broker_choice):
    """ Configure and return the quote asset. """
    return Asset(symbol="USD", asset_type="forex")

def get_benchmark_asset(broker_choice):
    if broker_choice == 'IB' or broker_choice == 'Alpaca':
        return Asset("SPY")
    if broker_choice == 'Kraken':
        return Asset("BTC", asset_type="crypto")
    
def run_strategy(strategy_class, is_live, broker_choice, start_date, end_date):
    """Run the specified trading strategy in live or backtesting mode."""
    if is_live:
        # Confirm broker configuration
        print(f"Configured to run in live mode with broker: {broker_choice}")
        confirm = input("Do you want to proceed? (yes/no): ").lower()
        if confirm != 'yes':
            print("Operation cancelled.")
            return

        # Setting up for live trading
        trader = Trader()
        broker = configure_broker()
        quote_asset = configure_quote()
        strategy = strategy_class(broker=broker, quote_asset = quote_asset)
        trader.add_strategy(strategy)
        trader.run_all()
    else:
        # Setting up for backtesting
        from lumibot.backtesting import PolygonDataBacktesting
        try:
            from credentials import POLYGON_CONFIG, BACKTESTING_CONFIG
        except ImportError as e:
            # Imprime un mensaje de error específico basado en la configuración faltante
            missing_config = str(e).split("named ")[-1]
            print(f"Error: La configuración '{missing_config}' no se encontró en 'credentials.py'.")
            exit(1)
        
        percent_fee = BACKTESTING_CONFIG['PERCENT_FEE']
        trading_fee = TradingFee(percent_fee=percent_fee)
        risk_free_rate = BACKTESTING_CONFIG['RISK_FREE_RATE']
        benchmark_asset = BACKTESTING_CONFIG['BENCHMARK']

        # Intenta obtener y validar las fechas de inicio y finalización
        try:
            start_date_str = BACKTESTING_CONFIG['START_DATE']
            end_date_str = BACKTESTING_CONFIG['END_DATE']
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            # Asegurarse de que la fecha de inicio no es posterior a la fecha de finalización
            if start_date >= end_date:
                raise ValueError("La fecha de inicio debe ser anterior a la fecha de finalización.")
        except KeyError:
            print("Error: Las fechas de inicio y finalización deben estar definidas en BACKTESTING_CONFIG.")
            exit(1)
        except ValueError as e:
            print(f"Error en las fechas de BACKTESTING_CONFIG: {e}")
            exit(1)

        print("Starting Backtest...")
        strategy_class.backtest(
            PolygonDataBacktesting,
            start_date,
            end_date,
            polygon_api_key=POLYGON_CONFIG["KEY"],
            polygon_has_paid_subscription=POLYGON_CONFIG["PAID"],
            benchmark_asset=benchmark_asset,
            risk_free_rate=risk_free_rate,
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            logfile=f'logs/{module_name}.log',
            parameters={}
        )

if __name__ == "__main__":
    # Main execution block
    args = parse_arguments()
    module_name = os.path.splitext(os.path.basename(args.strategy_file))[0]

    try:
        # Add the path to the strategy directory to sys.path
        strategy_relative_path = os.path.dirname(args.strategy_file)
        current_directory = os.getcwd()
        absolute_strategy_path = os.path.join(current_directory, strategy_relative_path)
        sys.path.append(absolute_strategy_path)

        # Dynamically import the strategy module and find the strategy class
        strategy_module = importlib.import_module(module_name)
        strategy_class = find_strategy_class(strategy_module)
        if not strategy_class:
            raise ImportError(f"No strategy class found in the module {module_name}.")
    except ImportError as e:
        print(f"Error importing strategy: {e}")
        exit(1)

    # Parse start and end dates for backtesting
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None

    # Run the specified strategy
    run_strategy(strategy_class, args.live, args.broker, start_date, end_date)
