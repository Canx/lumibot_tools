BACKTESTING_CONFIG = {
    "PERCENT_FEE": 0.005, # 0.5% de comisión por operación
    "RISK_FREE_RATE": 5.233, # Tasa libre de riesgo anual en porcentaje
    "BENCHMARK": "SPY", # Benchmark por defecto
    "START_DATE": "2023-01-01",  # YYYY-MM-DD o None para calcular automáticamente
    "END_DATE": "2024-01-01",  # YYYY-MM-DD o None para calcular automáticamente
}
ALPACA_CONFIG = {
    "API_KEY": "<API_KEY>",
    "API_SECRET": "<API_SECRET>",
    "PAPER": True
}
POLYGON_CONFIG = {
    "KEY": "<POLYGON KEY>"
}
TELEGRAM_CONFIG = {
    "TOKEN": "<TELEGRAM TOKEN>",
    "CHAT_ID": "<TELEGRAM CHAT ID>"
}
DISCORD_CONFIG = {
    "WEBHOOK": ""
}
