import streamlit as st
from datetime import datetime
import os
import joblib
import config
import credentials
from multiprocessing import Process, Queue
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


symbols = {
    "ACGL": ("Arch Capital Group", "2015-04-29", "2017-11-10"),
    "GOOG": ("Google", "2017-02-21", "2017-11-10"),
    "NVDA": ("NVIDIA Corporation", "2014-02-11", "2017-11-10"),
    "TSLA": ("Tesla Inc.", "2016-05-23", "2017-11-10"),
    "IEX": ("IEX Group", "2015-04-30", "2017-11-10"),
    "QCOM": ("Qualcomm Inc.", "2012-09-05", "2017-11-10"),
    "MCHP": ("Microchip Technology", "2012-12-07", "2017-11-10"),
    "AZN": ("AstraZeneca PLC", "2015-04-30", "2017-11-10")
}

def run_backtest(queue):
    try:
        from lumibot.strategies.strategy import Strategy
        from lumibot.brokers import Alpaca
        from lumibot.backtesting import YahooDataBacktesting
        from alpaca_trade_api import REST

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
                self.last_trade_symbol = None
                self.cash_at_risk = cash_at_risk
                self.api = REST(
                    base_url=credentials.BASE_URL,
                    key_id=credentials.api_key,
                    secret_key=credentials.api_secret
                )

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
                        continue
                return best_symbol, best_pct_change, best_last_price

            def on_trading_iteration(self):
                best_symbol, pct_change, last_price = self.select_best_stock()
                if not best_symbol:
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

                if self.last_trade and (self.last_trade != order_type or 
                                      self.last_trade_symbol != best_symbol):
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

        symbol_dates = {sym: (symbols[sym][1], symbols[sym][2]) for sym in symbols}
        backtest_start = min(datetime.strptime(dates[0], "%Y-%m-%d") 
                            for sym, dates in symbol_dates.items())
        backtest_end = max(datetime.strptime(dates[1], "%Y-%m-%d") 
                          for sym, dates in symbol_dates.items())

        broker = Alpaca(config.ALPACA_CREDS)
        strategy = ARIMAStrategy(
            name="arima_backtest_multi_stock",
            broker=broker
        )
        
        results = strategy.backtest(
            YahooDataBacktesting,
            backtest_start,
            backtest_end,
            show_tearsheet = False,
            save_tearsheet = False
        )
        tearsheet_html = results.get_tearsheet()
        encoded_html = base64.b64encode(tearsheet_html.encode()).decode()
        queue.put({
            "status": "success",
            "tearsheet": encoded_html,
            "stats": results.get_stats()
        })

    except Exception as e:
        queue.put({"status": "error", "message": str(e)})

ef display_tearsheet(encoded_html):
    decoded_html = base64.b64decode(encoded_html).decode()
    components.html(decoded_html, height=800, scrolling=True)


# Streamlit UI
st.title("SuperAlgos.ai - ARIMA Trading Bot")

st.write("""
This trading bot uses ARIMA models to forecast stock prices and automatically trade the stock 
with the strongest predicted price movement. The strategy implements bracket orders with 
take-profit and stop-loss levels to manage risk.
""")

st.header("Trained Assets")
cols = st.columns(4)
for idx, (symbol, (name, start, end)) in enumerate(symbols.items()):
    with cols[idx % 4]:
        st.subheader(f"{name}")
        st.caption(f"**Ticker:** {symbol}")
        st.caption(f"**Training Period:**\n{start} to {end}")

# Bot information
st.write("""  
**Risk Parameters:**  
- 50% cash-at-risk per trade  
- Daily rebalancing  
- Automatic position closure on new signals   
""")

# Session state initialization
if 'backtest_process' not in st.session_state:
    st.session_state.backtest_process = None
if 'results_queue' not in st.session_state:
    st.session_state.results_queue = Queue()
if 'backtest_status' not in st.session_state:
    st.session_state.backtest_status = "ready"

if st.button("🚀 Execute Backtest", use_container_width=True):

    missing_models = []
    for symbol in symbols:
        model_file = f"./Models/arima_model_{symbol.lower()}.pkl"
        if not os.path.exists(model_file):
            missing_models.append(symbol)
    if missing_models:
        st.error(f"Missing models for: {', '.join(missing_models)}")
    else:
        st.session_state.results_queue = Queue()
        st.session_state.backtest_process = Process(
            target=run_backtest,
            args=(st.session_state.results_queue,)
        )
        st.session_state.backtest_process.start()
        st.session_state.backtest_status = "running"
status_placeholder = st.empty()

if st.session_state.backtest_status == "running":
    with status_placeholder.container():
        with st.spinner(""):
            st.markdown(
                f"<div style='color: white; text-align: center; margin: 2rem 0;'>"
                f"⚡ Processing backtest for all trained assets...<br>"
                f"Estimated completion in 4-5 minutes"
                f"</div>", 
                unsafe_allow_html=True
            )
            while st.session_state.backtest_process.is_alive():
                time.sleep(1)
            st.session_state.backtest_process.join()
            if not st.session_state.results_queue.empty():
                result = st.session_state.results_queue.get()
                if result["status"] == "success":
                    status_placeholder.success("✅ Backtest completed successfully!")
                    st.balloons()
                    display_tearsheet(result["tearsheet"])
                else:
                    status_placeholder.error(f"❌ Backtest failed: {result['message']}")
                st.session_state.backtest_status = "ready"
            else:
                status_placeholder.error("⚠️ Backtest process terminated unexpectedly")
                st.session_state.backtest_status = "ready"

st.markdown("---")
st.subheader("About the ARIMA Trading Strategy")
st.write("""
The ARIMA trading strategy works as follows:

1. **Stock Selection**: The bot analyzes all selected stocks and chooses the one with the strongest 
   predicted price movement (either up or down).

2. **Position Sizing**: The bot calculates the appropriate position size based on the available cash 
   and configured risk level.

3. **Order Execution**: 
   - If the price is predicted to rise, the bot places a buy order
   - If the price is predicted to fall, the bot places a sell order
   
4. **Risk Management**: Each order includes:
   - Take-profit level (20% gain for long positions, 20% drop for short positions)
   - Stop-loss level (5% loss for long positions, 5% gain for short positions)

5. **Position Management**: The bot closes existing positions before opening new ones in a different 
   direction or on a different stock.
""")
