from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from alpaca_trade_api import REST
from feature_engineering import add_features,exogenous_features
from datetime import datetime
import joblib
import os
import config
import credentials
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


symbols = {
    "ACGL": ("2015-04-29", "2017-11-10"),
    "GOOG": ("2017-02-21", "2017-11-10"),
    "NVDA": ("2014-02-11", "2017-11-10"),
    "TSLA": ("2016-05-23", "2017-11-10"),
    "IEX": ("2015-04-30", "2017-11-10"),
    "QCOM": ("2012-09-05", "2017-11-10"),
    "MCHP": ("2012-12-07", "2017-11-10"),
    "AZN": ("2015-04-30", "2017-11-10")
}

models = {}
for symbol in symbols:
    model_filename = f"./Models/arima_model_{symbol.lower()}.pkl"
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    models[symbol] = joblib.load(model_filename)

class ARIMAStrategy(Strategy):
    def initialize(self, cash_at_risk: float = 0.5):
        self.symbols = list(symbols.keys())
        self.models = models
        self.sleeptime = '24H'
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=credentials.BASE_URL, key_id=credentials.api_key, secret_key=credentials.api_secret)

    def position_sizing(self, symbol):
        cash = self.get_cash()
        last_price = self.get_last_price(symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def forecast_pct_change(self, symbol):
        model = self.models[symbol]
        last_price = self.get_last_price(symbol)
        forecast = model.predict(n_periods=5)
        predicted_price = forecast[-1]
        pct_change = (predicted_price - last_price) / last_price
        return pct_change, last_price

    def select_best_stock(self):
        best_symbol = None
        best_abs_change = 0
        best_pct_change = 0
        best_last_price = 0

        for symbol in self.symbols:
            try:
                pct_change, last_price = self.forecast_pct_change(symbol)
                if abs(pct_change) > best_abs_change:
                    best_abs_change = abs(pct_change)
                    best_symbol = symbol
                    best_pct_change = pct_change
                    best_last_price = last_price
            except Exception as e:
                print(f"Error forecasting {symbol}: {e}")
                continue

        return best_symbol, best_pct_change, best_last_price

    def on_trading_iteration(self):
        best_symbol, pct_change, last_price = self.select_best_stock()
        if best_symbol is None or last_price == 0:
            return

        cash, last_price, quantity = self.position_sizing(best_symbol)
        if cash < last_price or quantity < 1:
            return

        if pct_change > 0:
            order_type = "buy"
            tp_price = last_price * 1.20
            sl_price = last_price * 0.95
        else:
            order_type = "sell"
            tp_price = last_price * 0.80
            sl_price = last_price * 1.05

        
        if self.last_trade and (self.last_trade != order_type or self.last_trade_symbol != best_symbol):
            self.sell_all()

        order = self.create_order(
            best_symbol,
            quantity,
            order_type,
            type="bracket",
            take_profit_price=tp_price,
            stop_loss_price=sl_price
        )
        self.submit_order(order)
        self.last_trade = order_type
        self.last_trade_symbol = best_symbol


backtest_start = min(datetime.strptime(v[0], "%Y-%m-%d") for v in symbols.values())
backtest_end = max(datetime.strptime(v[1], "%Y-%m-%d") for v in symbols.values())

broker = Alpaca(config.ALPACA_CREDS)
strategy = ARIMAStrategy(
    name="arima_backtest_multi_stock",
    broker=broker
)
strategy.backtest(
    YahooDataBacktesting,
    backtest_start,
    backtest_end
)