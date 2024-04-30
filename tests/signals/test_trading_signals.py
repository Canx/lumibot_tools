import pytest
from unittest.mock import MagicMock, patch
from lumibot_tools.signals import Signals  # Asume que tu clase está en signals.py
import pandas as pd

@pytest.fixture
def mock_strategy():
    return MagicMock()

@pytest.fixture
def signals(mock_strategy):
    return Signals(strategy=mock_strategy)

# Test para un nuevo máximo
def test_new_price_high(signals, mock_strategy):
    mock_strategy.get_historical_prices.return_value = MagicMock(df=pd.DataFrame({
        'close': [100, 105, 110, 115, 120]  # Ejemplo de precios de cierre
    }))
    result = signals.new_price_high_or_low('AAPL', days=5, type='high')
    assert result == True
    signals.strategy.log_message.assert_called_with("AAPL has reached a new 5-day high at 120.")

# Test para cuando el precio más reciente no es un nuevo máximo
def test_new_price_not_high_enough(signals, mock_strategy):
    mock_strategy.get_historical_prices.return_value = MagicMock(df=pd.DataFrame({
        'close': [120, 119, 118, 117, 116]  # Los precios están bajando, sin nuevo máximo
    }))
    result = signals.new_price_high_or_low('AAPL', days=5, type='high')
    assert result == False
    signals.strategy.log_message.assert_called_with("No high condition met for AAPL")

# Test para un nuevo mínimo
def test_new_price_low(signals, mock_strategy):
    mock_strategy.get_historical_prices.return_value = MagicMock(df=pd.DataFrame({
        'close': [120, 110, 105, 102, 100]  # El último precio es un nuevo mínimo
    }))
    result = signals.new_price_high_or_low('AAPL', days=5, type='low')
    assert result == True
    signals.strategy.log_message.assert_called_with("AAPL has reached a new 5-day low at 100.")

# Test para manejar datos faltantes
def test_missing_data(signals, mock_strategy):
    mock_strategy.get_historical_prices.return_value = MagicMock(df=pd.DataFrame({
        'open': [100, 105, 110, 115, 120]  # Sin columna 'close'
    }))
    result = signals.new_price_high_or_low('AAPL', days=5, type='high')
    assert result == False
    signals.strategy.log_message.assert_called_with("Error: No 'close' price available for AAPL")

# Test para precios constantes
def test_constant_prices(signals, mock_strategy):
    mock_strategy.get_historical_prices.return_value = MagicMock(df=pd.DataFrame({
        'close': [100, 100, 100, 100, 100]  # Todos los precios son iguales
    }))
    result = signals.new_price_high_or_low('AAPL', days=5, type='high')
    assert result == False  # Depende si consideras esto como un nuevo máximo o no
    signals.strategy.log_message.assert_called_with("No high condition met for AAPL")


