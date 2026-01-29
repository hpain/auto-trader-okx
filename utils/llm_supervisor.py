import os
import time
import json
import logging
import threading
import asyncio
import traceback
from datetime import datetime
import sys

# Ensure utils is importable if run directly (though this is a library)
sys.path.append(os.getcwd())

from utils.llm_client import LLMClient
from utils.llm_client_genai import GeminiClient
from exchange.factory import ExchangeFactory

class LLMSupervisor:
    """
    LLM Supervisor running in a separate thread.
    Fetches market data periodically, queries LLM, and updates 'data/market_context.json'.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger("LLMSupervisor")
        self.config = config or {}
        
        # Load Env Config
        self.api_key = os.getenv("LLM_API_KEY") 
        # Base URL is less relevant for Native SDK but kept for compat
        self.base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        
        self.symbol = "BTC-USDT" 
        self.timeframe = "4H"
        self.update_interval = 900 # 15 minutes
        self.context_file = "data/market_context.json"
        
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        if not self.api_key:
            self.logger.warning("⚠️ LLM_API_KEY not set. Supervisor will be disabled or fail.")

    def start(self):
        """Start the supervisor daemon thread."""
        if self.running:
            self.logger.warning("Supervisor already running.")
            return

        self.logger.info(f"🤖 Starting LLM Supervisor (Threaded)...")
        self.logger.info(f"   Model: {self.model_name}")
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.running = True

    def stop(self):
        """Stop the supervisor thread."""
        if not self.running:
            return
            
        self.logger.info("Stopping LLM Supervisor...")
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.running = False
        self.logger.info("LLM Supervisor Stopped.")

    def _run_loop(self):
        """The main loop running in the thread."""
        # Use Native Gemini Client if model name contains 'gemini'
        if 'gemini' in self.model_name.lower():
            self.logger.info("Using Native Gemini SDK Client.")
            llm = GeminiClient(api_key=self.api_key, model=self.model_name)
        else:
            self.logger.info(f"Using Generic OpenAI-compatible Client ({self.base_url}).")
            llm = LLMClient(api_key=self.api_key, base_url=self.base_url, model=self.model_name)
        
        while not self.stop_event.is_set():
            try:
                self.logger.info("Fetching market data...")
                # We use specific asyncio loop for this thread if needed, 
                # but asyncio.run() create a new event loop for the call which is fine for synchronous thread.
                data_summary = asyncio.run(self._fetch_market_data())
                
                if not data_summary:
                    self.logger.warning("No data received. Waiting...")
                    if self.stop_event.wait(60): # Wait 60s or stop
                        break
                    continue
                
                self.logger.debug(f"Market Data: {data_summary}")
                
                # Queries LLM
                self._query_and_update(llm, data_summary)
                
            except Exception as e:
                self.logger.error(f"Supervisor loop error: {e}")
                # traceback.print_exc()
            
            # Wait for next interval
            if self.stop_event.wait(self.update_interval):
                break

    async def _fetch_market_data(self):
        """Fetch market data using ccxt (async, created new loop)."""
        # Use OKX for market data (same exchange we trade on)
        # sandbox=False to get real public market data (no auth needed)
        exchange = await ExchangeFactory.create_exchange("okx", sandbox=False)
        try:
            # Fetch OHLCV data
            ohlcv = await exchange.fetch_candles(self.symbol, self.timeframe, limit=100)
            if ohlcv is None or ohlcv.empty:
                return None
                
            df = ohlcv.copy()
            df['ret'] = df['close'].pct_change()
            df['vol_24'] = df['ret'].rolling(24).std()
            df['ma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = 100 - (100 / (1 + df['ret'].rolling(14).apply(lambda x: x[x>0].mean() / abs(x[x<0].mean()) if len(x[x<0]) > 0 else 1)))
            
            latest = df.iloc[-1]
            
            summary = {
                "symbol": self.symbol,
                "price": float(latest['close']),
                "change_24h_pct": float((latest['close'] / df.iloc[-6]['close'] - 1) * 100),
                "volatility_atr_proxy": float(latest['vol_24'] * 100),
                "trend_4h": "bullish" if latest['close'] > latest['ma_50'] else "bearish",
                "rsi_14": float(latest['rsi']),
                "last_5_bars_close": df['close'].tail(5).tolist()
            }
            
            # ----------------------------------------------------------------
            # NEW: Fetch Derivatives Data for enhanced risk assessment
            # ----------------------------------------------------------------
            derivatives = {}
            
            try:
                # 1. Funding Rate
                funding_df = await exchange.fetch_funding_rates(self.symbol, "1H", limit=24)
                if funding_df is not None and not funding_df.empty:
                    latest_funding = float(funding_df['fundingRate'].iloc[-1])
                    avg_funding_24h = float(funding_df['fundingRate'].mean())
                    derivatives["funding_rate"] = latest_funding
                    derivatives["funding_rate_24h_avg"] = avg_funding_24h
                    derivatives["funding_abnormal"] = abs(latest_funding) > 0.0003  # >0.03% is high
            except Exception as e:
                self.logger.debug(f"Funding rate fetch failed: {e}")
            
            try:
                # 2. Open Interest
                oi_df = await exchange.fetch_open_interest(self.symbol, "1H", limit=24)
                if oi_df is not None and not oi_df.empty:
                    latest_oi = float(oi_df['openInterest'].iloc[-1])
                    prev_oi = float(oi_df['openInterest'].iloc[0])
                    oi_change_pct = ((latest_oi / prev_oi) - 1) * 100 if prev_oi > 0 else 0
                    derivatives["open_interest"] = latest_oi
                    derivatives["oi_change_24h_pct"] = round(oi_change_pct, 2)
            except Exception as e:
                self.logger.debug(f"Open interest fetch failed: {e}")
            
            try:
                # 3. Long/Short Ratio
                ls_df = await exchange.fetch_long_short_ratio(self.symbol, "1H", limit=24)
                if ls_df is not None and not ls_df.empty:
                    col_name = 'longShortRatio' if 'longShortRatio' in ls_df.columns else ls_df.columns[-1]
                    latest_ls = float(ls_df[col_name].iloc[-1])
                    derivatives["long_short_ratio"] = round(latest_ls, 3)
                    derivatives["crowd_sentiment"] = "extremely_long" if latest_ls > 2.0 else \
                                                     "long_biased" if latest_ls > 1.2 else \
                                                     "neutral" if latest_ls > 0.8 else \
                                                     "short_biased" if latest_ls > 0.5 else "extremely_short"
            except Exception as e:
                self.logger.debug(f"Long/short ratio fetch failed: {e}")
            
            if derivatives:
                summary["derivatives"] = derivatives
            
            # ----------------------------------------------------------------
            # NEW: Load ML Model Performance Feedback
            # ----------------------------------------------------------------
            ml_performance = self._load_ml_performance()
            if ml_performance:
                summary["ml_model_performance"] = ml_performance
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Fetch error: {e}")
            return None
        finally:
            try:
                await exchange.close()
            except:
                pass
    
    def _load_ml_performance(self):
        """Load ML model performance feedback from file."""
        try:
            ml_file = "data/ml_performance.json"
            if not os.path.exists(ml_file):
                return None
            
            # Check file age (ignore if >1 hour old)
            mtime = os.path.getmtime(ml_file)
            if time.time() - mtime > 3600:
                return None
            
            with open(ml_file, 'r') as f:
                data = json.load(f)
            return data.get('summary', None)
        except Exception as e:
            self.logger.debug(f"Failed to load ML performance: {e}")
            return None

    def _query_and_update(self, llm, data_summary):
        """Construct prompt and query LLM."""
        system_prompt = """You are a senior crypto trading risk manager. 
        Your job is to analyze market conditions and control the risk of a high-frequency scalping bot.
        
        You will receive:
        1. Price data with technical indicators (trend, RSI, volatility)
        2. Derivatives data (funding rate, open interest, long/short ratio)
        3. ML model performance metrics (win rate, confidence trend)
        
        Key Risk Rules:
        - If funding rate is abnormal (>0.03% or <-0.03%), increase caution
        - If crowd sentiment is extremely biased, consider contrarian bias
        - If ML model win rate is declining, reduce risk_multiplier
        - If volatility is extreme (>3%) or trend is unclear, go defensive
        
        Output JSON format only:
        {
            "regime": "string (trending_up, trending_down, ranging, high_volatility, crash)",
            "risk_multiplier": "float (0.0 to 1.5, where 1.0=normal, 0.5=defensive, 0.0=stop)",
            "bias": "string (long, short, neutral)",
            "tp_sl_suggestion": {"tp_pct": 0.02, "sl_pct": 0.01},
            "reasoning": "string (concise explanation, max 100 words)"
        }
        """
        
        # Build enhanced context
        derivatives_info = data_summary.get('derivatives', {})
        ml_info = data_summary.get('ml_model_performance', {})
        
        user_prompt = f"""
        === MARKET STATUS ({datetime.utcnow()} UTC) ===
        
        📈 Price Data:
        - Symbol: {data_summary.get('symbol')}
        - Price: ${data_summary.get('price', 0):,.2f}
        - 24H Change: {data_summary.get('change_24h_pct', 0):.2f}%
        - Volatility: {data_summary.get('volatility_atr_proxy', 0):.2f}%
        - Trend (4H): {data_summary.get('trend_4h', 'unknown')}
        - RSI(14): {data_summary.get('rsi_14', 50):.1f}

        📊 Derivatives Indicators:
        - Funding Rate: {derivatives_info.get('funding_rate', 'N/A')}
        - Funding Abnormal: {derivatives_info.get('funding_abnormal', False)}
        - Open Interest Change (24h): {derivatives_info.get('oi_change_24h_pct', 'N/A')}%
        - Long/Short Ratio: {derivatives_info.get('long_short_ratio', 'N/A')}
        - Crowd Sentiment: {derivatives_info.get('crowd_sentiment', 'unknown')}

        🤖 ML Model Performance:
        - Win Rate: {ml_info.get('win_rate', 'N/A')}
        - Avg Confidence: {ml_info.get('avg_confidence', 'N/A')}
        - Performance Trend: {ml_info.get('trend', 'unknown')}
        
        === TASK ===
        1. Analyze all data holistically
        2. Determine market regime and appropriate risk level
        3. If ML model is underperforming (declining trend or <40% win rate), reduce exposure
        4. Suggest appropriate TP/SL percentages based on volatility
        5. Provide concise reasoning
        """
        
        decision = llm.query_json(user_prompt, system_prompt)
        
        if decision:
            decision['timestamp'] = datetime.utcnow().isoformat()
            
            # Atomic write
            temp_file = self.context_file + ".tmp"
            try:
                os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
                with open(temp_file, 'w') as f:
                    json.dump(decision, f, indent=2)
                os.replace(temp_file, self.context_file)
                self.logger.info(f"✅ LLM Update: Risk={decision.get('risk_multiplier')}, Bias={decision.get('bias')}")
            except Exception as e:
                self.logger.error(f"Failed to write context file: {e}")
        else:
            self.logger.error("LLM returned empty response.")
