from fastapi import FastAPI, HTTPException
import logging
import talib as ta
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
import fmpsdk
import pandas as pd
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

load_dotenv()
apikey = os.environ.get("API_KEY")

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(filename="./example.log", encoding="utf-8", level=logging.DEBUG)

# Mapping of indicators to their required inputs and default parameters
INDICATOR_INFO = {
    'SMA': {'inputs': ['adjClose'], 'defaults': {'timeperiod': 14}},
    'EMA': {'inputs': ['adjClose'], 'defaults': {'timeperiod': 14}},
    'RSI': {'inputs': ['adjClose'], 'defaults': {'timeperiod': 14}},
    'MACD': {'inputs': ['adjClose'], 'defaults': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}},
    'CCI': {'inputs': ['high', 'low', 'adjClose'], 'defaults': {'timeperiod': 14}},
    'STOCH': {'inputs': ['high', 'low', 'adjClose'], 'defaults': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3}},
    'ATR': {'inputs': ['high', 'low', 'adjClose'], 'defaults': {'timeperiod': 14}},
    # Add more indicators as needed
}

class IndicatorConfig(BaseModel):
    name: str
    params: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None  # Threshold for buy/sell signals

class StrategyConfig(BaseModel):
    symbol: str
    start_date: Optional[str] = Field(None, pattern="^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, pattern="^\d{4}-\d{2}-\d{2}$")
    starting_balance: float = 10000.0
    buy_indicators: List[IndicatorConfig]
    sell_indicator: IndicatorConfig
    baseline_indicator: IndicatorConfig

@app.post("/currency")
async def run_strategy(config: StrategyConfig):
    try:
        symbol = config.symbol

        # Parse dates
        if config.start_date:
            from_date = datetime.strptime(config.start_date, '%Y-%m-%d')
        else:
            from_date = None
        if config.end_date:
            to_date = datetime.strptime(config.end_date, '%Y-%m-%d')
        else:
            to_date = None

        # Fetch historical data
        data = fmpsdk.historical_price_full(apikey=apikey, symbol=symbol, from_date=from_date, to_date=to_date)
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given parameters")

        # Reset index
        df = df.reset_index(drop=True)

        # Ensure required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'adjClose', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan  # Fill missing columns with NaN

        # Compute ATR
        atr_info = INDICATOR_INFO['ATR']
        atr_function = getattr(ta, 'ATR')
        atr_inputs = [df[col] for col in atr_info['inputs']]
        atr_params = atr_info['defaults']
        df['ATR'] = atr_function(*atr_inputs, **atr_params)

        # Compute Baseline Indicator
        baseline_name = config.baseline_indicator.name.upper()
        baseline_info = INDICATOR_INFO.get(baseline_name)
        if not baseline_info:
            raise HTTPException(status_code=400, detail=f"Unsupported baseline indicator: {baseline_name}")

        baseline_function = getattr(ta, baseline_name)
        baseline_inputs = [df[col] for col in baseline_info['inputs']]
        baseline_params = baseline_info['defaults']
        # Override defaults if params provided
        if config.baseline_indicator.params:
            baseline_params.update(config.baseline_indicator.params)
        df['Baseline'] = baseline_function(*baseline_inputs, **baseline_params)

        # Compute Buy Indicators
        for idx, indicator in enumerate(config.buy_indicators):
            name = indicator.name.upper()
            indicator_info = INDICATOR_INFO.get(name)
            if not indicator_info:
                raise HTTPException(status_code=400, detail=f"Unsupported buy indicator: {name}")

            function = getattr(ta, name)
            inputs = [df[col] for col in indicator_info['inputs']]
            params = indicator_info['defaults'].copy()
            if indicator.params:
                params.update(indicator.params)
            threshold = indicator.threshold
            indicator_column = f"Buy_Indicator_{idx+1}"

            # Compute indicator
            result = function(*inputs, **params)

            # Handle indicators that return multiple outputs
            if isinstance(result, tuple):
                df[indicator_column] = result[0]
            else:
                df[indicator_column] = result

            df[f"Buy_Threshold_{idx+1}"] = threshold

        # Compute Sell Indicator
        sell_name = config.sell_indicator.name.upper()
        sell_indicator_info = INDICATOR_INFO.get(sell_name)
        if not sell_indicator_info:
            raise HTTPException(status_code=400, detail=f"Unsupported sell indicator: {sell_name}")

        sell_function = getattr(ta, sell_name)
        sell_inputs = [df[col] for col in sell_indicator_info['inputs']]
        sell_params = sell_indicator_info['defaults'].copy()
        if config.sell_indicator.params:
            sell_params.update(config.sell_indicator.params)
        sell_threshold = config.sell_indicator.threshold
        sell_indicator_column = "Sell_Indicator"

        sell_result = sell_function(*sell_inputs, **sell_params)
        if isinstance(sell_result, tuple):
            df[sell_indicator_column] = sell_result[0]
        else:
            df[sell_indicator_column] = sell_result
        df["Sell_Threshold"] = sell_threshold

        # Initialize variables
        balance = config.starting_balance
        position = None  # No open position initially
        equity_curve = []
        total_equity = balance
        orders = []  # List to store orders

        for i in range(1, len(df)):
            date = df['date'].iloc[i]
            close = df['adjClose'].iloc[i]
            atr_value = df['ATR'].iloc[i]
            baseline_value = df['Baseline'].iloc[i]

            # Check for NaN values
            if any(np.isnan([atr_value, baseline_value, close])):
                continue  # Skip if any critical value is NaN

            # Buy conditions
            buy_signals = []
            for idx, indicator in enumerate(config.buy_indicators):
                indicator_value = df[f"Buy_Indicator_{idx+1}"].iloc[i]
                indicator_prev = df[f"Buy_Indicator_{idx+1}"].iloc[i - 1]
                threshold = indicator.threshold

                if np.isnan(indicator_value) or np.isnan(indicator_prev):
                    buy_signals.append(False)
                    continue

                name = indicator.name.upper()
                if threshold is not None:
                    # For indicators with thresholds (e.g., RSI crossing above a level)
                    buy_signal = (indicator_prev < threshold) and (indicator_value > threshold)
                else:
                    # For moving averages or other indicators without thresholds
                    # Example: Simple moving average crossover (price crossing above the SMA)
                    buy_signal = close > indicator_value and close <= indicator_prev
                buy_signals.append(buy_signal)

            price_above_baseline = close > baseline_value

            # Sell conditions
            sell_indicator_value = df[sell_indicator_column].iloc[i]
            sell_indicator_prev = df[sell_indicator_column].iloc[i - 1]
            if np.isnan(sell_indicator_value) or np.isnan(sell_indicator_prev):
                sell_signal = False
            else:
                if sell_threshold is not None:
                    sell_signal = (sell_indicator_prev > sell_threshold) and (sell_indicator_value < sell_threshold)
                else:
                    # Example: Simple moving average crossover (price crossing below the EMA)
                    sell_signal = close < sell_indicator_value and close >= sell_indicator_prev
            price_below_baseline = close < baseline_value

            if position is None:
                # Check for buy signal
                if any(buy_signals) and price_above_baseline:
                    # Calculate position size based on ATR
                    risk_per_trade = 0.01 * balance  # 1% of current balance
                    stop_loss_price = close - (2 * atr_value)
                    stop_loss_amount = 2 * atr_value
                    position_size = risk_per_trade / stop_loss_amount
                    if position_size <= 0:
                        continue  # Cannot open a position

                    cost = position_size * close
                    if cost > balance:
                        continue  # Not enough balance to open position

                    balance -= cost  # Deduct cost from balance
                    position = {
                        'entry_price': close,
                        'position_size': position_size,
                        'stop_loss': stop_loss_price,
                        'take_profit': close + (2 * atr_value),
                        'entry_date': date
                    }
                    logger.info(f"Opened position on {date}: size {position_size} at price {close}")

                    # Record BUY order
                    orders.append({
                        'date': date,
                        'type': 'BUY',
                        'position_size': position_size,
                        'price': close,
                        'amount': cost,
                        'reason': 'Entry'
                    })

            else:
                # Check for exit conditions
                if close <= position['stop_loss']:
                    proceeds = position['position_size'] * close
                    balance += proceeds
                    logger.info(f"Position closed at stop-loss on {date}: size {position['position_size']} at price {close}")

                    # Record SELL order
                    orders.append({
                        'date': date,
                        'type': 'SELL',
                        'position_size': position['position_size'],
                        'price': close,
                        'amount': proceeds,
                        'reason': 'Stop-Loss'
                    })

                    position = None
                elif close >= position['take_profit']:
                    proceeds = position['position_size'] * close
                    balance += proceeds
                    logger.info(f"Position closed at take-profit on {date}: size {position['position_size']} at price {close}")

                    # Record SELL order
                    orders.append({
                        'date': date,
                        'type': 'SELL',
                        'position_size': position['position_size'],
                        'price': close,
                        'amount': proceeds,
                        'reason': 'Take-Profit'
                    })

                    position = None
                elif sell_signal or price_below_baseline:
                    proceeds = position['position_size'] * close
                    balance += proceeds
                    logger.info(f"Position closed at sell signal on {date}: size {position['position_size']} at price {close}")

                    # Record SELL order
                    orders.append({
                        'date': date,
                        'type': 'SELL',
                        'position_size': position['position_size'],
                        'price': close,
                        'amount': proceeds,
                        'reason': 'Sell Signal'
                    })

                    position = None

            # Record equity
            if position:
                position_value = position['position_size'] * close
                total_equity = balance + position_value
            else:
                total_equity = balance

            equity_curve.append({'date': str(date), 'equity': total_equity})

        # Calculate total return
        total_return = ((total_equity - config.starting_balance) / config.starting_balance) * 100
        total_profit = total_equity - config.starting_balance
        logger.info(f"Total return: {total_return:.2f}%")

        return {
            'total_return': total_return,
            'total_profit': total_profit,
            'equity_curve': equity_curve,
            'orders': orders  # Include orders in the response
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
