class TrendStrength:


    @classmethod
    def calculate_max_min_based(signals, symbol, lookback_windows, weight_per_signal=0.1, start_date=None, end_date=None):
        def within_date_range(signal_date):
            if start_date and signal_date < start_date:
                return False
            if end_date and signal_date > end_date:
                return False
            return True

        high_signals = [s for s in signals.signal_history[symbol]['high'] if s['length'] in lookback_windows and s['condition_met'] and within_date_range(s['date'])]
        low_signals = [s for s in signals.signal_history[symbol]['low'] if s['length'] in lookback_windows and s['condition_met'] and within_date_range(s['date'])]

        positive_signals = len(high_signals)
        negative_signals = len(low_signals)

        trend_strength = positive_signals * weight_per_signal - negative_signals * weight_per_signal

        # Normalizar trend_strength para que esté en el rango de -1 a 1
        normalized_strength = max(-1, min(1, trend_strength))

        return normalized_strength