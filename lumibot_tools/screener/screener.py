from polygon import RESTClient
import pandas as pd
import datetime

class Screener:
    def __init__(self, api_key):
        self.api_key = api_key
        self.symbols = []
        self.data = {}

    def fetch_symbols(self):
        # Inicializa el cliente de Polygon
        with RESTClient(self.api_key) as client:
            # Consulta todos los tickers activos en el mercado de valores
            response = client.list_tickers(market="stocks", active=True)
            # Filtra para mantener solo los que están activos y son de tipo CS (common stocks)
            self.symbols = [ticker.ticker for ticker in response if ticker.active and ticker.type == 'CS']
            print(f"Fetched {len(self.symbols)} active stock symbols from the market.")

    def fetch_data(self, start_date, end_date, interval="1d"):
        # Implementar la función get_price_data_from_polygon como antes para descargar los datos
        for symbol in self.symbols:
            try:
                historical_data = get_price_data_from_polygon(
                    api_key=self.api_key,
                    asset=symbol,
                    start=start_date,
                    end=end_date,
                    timespan=interval
                )
                self.data[symbol] = historical_data
                print(f"Data fetched for {symbol}")
            except Exception as e:
                print(f"Failed to fetch data for {symbol}: {e}")

# Uso de la clase
api_key = "tu_api_key_aquí"
screener = Screener(api_key)
screener.fetch_symbols()
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime.now()
screener.fetch_data(start_date, end_date)
