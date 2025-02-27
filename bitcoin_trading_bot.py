import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone, timedelta
import pytz
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("btc_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BTC_RSI_Bot")

class BTCRsiOversoldBot:
    def __init__(self, 
                 rsi_period: int = 14, 
                 rsi_threshold: float = 30.0,
                 volume_multiplier: float = 1.2,
                 check_interval: int = 60,  # seconds
                 optimal_hours: List[int] = [1, 7, 5, 16, 13],
                 prioritize_time_windows: bool = True):
        """
        Initialize the Bitcoin RSI Oversold Bounce trading bot.
        
        Args:
            rsi_period: The period for RSI calculation (default: 14)
            rsi_threshold: The RSI threshold for oversold condition (default: 30.0)
            volume_multiplier: Minimum volume relative to average (default: 1.2)
            check_interval: How often to check for signals in seconds (default: 60)
            optimal_hours: List of hours (UTC) with highest win rates (default: [1, 7, 5, 16, 13])
            prioritize_time_windows: Whether to only alert during optimal hours (default: True)
        """
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
        self.check_interval = check_interval
        self.optimal_hours = optimal_hours
        self.prioritize_time_windows = prioritize_time_windows
        
        # Data storage
        self.price_data = pd.DataFrame()
        self.last_alert_time = None
        
        logger.info(f"Bot initialized with: RSI Period={rsi_period}, RSI Threshold={rsi_threshold}, "
                    f"Volume Multiplier={volume_multiplier}")
    
    def fetch_binance_data(self) -> pd.DataFrame:
        """
        Fetch Bitcoin 1-minute data from Binance API.
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info("Fetching 1-minute data from Binance API...")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": 1000  # Last 1000 1-minute candles (maximum allowed)
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Error fetching data from Binance: {response.status_code}, {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            # Convert to DataFrame with proper column names
            # Binance klines format: [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, 
            #                         Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Keep only the columns we need
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(df)} data points from Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rsi(self, data: pd.DataFrame, column: str = "close") -> np.ndarray:
        """
        Calculate the Relative Strength Index (RSI) for the given data.
        
        Args:
            data: DataFrame with price data
            column: Column name to use for RSI calculation (default: "close")
            
        Returns:
            NumPy array with RSI values
        """
        delta = data[column].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over RSI period
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    def calculate_relative_volume(self, data: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """
        Calculate the relative volume compared to average.
        
        Args:
            data: DataFrame with volume data
            lookback: Number of periods to use for average calculation (default: 20)
            
        Returns:
            NumPy array with relative volume values
        """
        avg_volume = data["volume"].rolling(window=lookback).mean()
        relative_volume = data["volume"] / avg_volume
        
        return relative_volume.values
    
    def is_optimal_trading_hour(self) -> bool:
        """
        Check if current UTC hour is in the optimal trading window.
        
        Returns:
            Boolean indicating if current hour is optimal for trading
        """
        current_hour = datetime.now(timezone.utc).hour
        return current_hour in self.optimal_hours
    
    def check_entry_conditions(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Check if entry conditions for the RSI Oversold Bounce pattern are met.
        
        Args:
            data: DataFrame with price, RSI, and volume data
            
        Returns:
            Dictionary with signal details if conditions are met, None otherwise
        """
        # Need at least rsi_period + 1 data points
        if len(data) < self.rsi_period + 1:
            return None
        
        # Calculate indicators
        data.loc[:, "rsi"] = self.calculate_rsi(data)
        data.loc[:, "relative_volume"] = self.calculate_relative_volume(data)
        
        # Check latest data point for entry conditions
        latest_idx = len(data) - 1
        prev_idx = latest_idx - 1
        
        # Skip if we don't have enough data for RSI
        if np.isnan(data.loc[latest_idx, "rsi"]) or np.isnan(data.loc[prev_idx, "rsi"]):
            return None
        
        # 1. RSI below threshold (oversold)
        is_oversold = data.loc[prev_idx, "rsi"] < self.rsi_threshold
        
        # 2. RSI turning up
        is_rsi_turning_up = data.loc[latest_idx, "rsi"] > data.loc[prev_idx, "rsi"]
        
        # 3. Bullish candle (close > open)
        is_bullish = data.loc[latest_idx, "close"] > data.loc[latest_idx, "open"]
        
        # 4. Above average volume
        is_high_volume = (
            not np.isnan(data.loc[latest_idx, "relative_volume"]) and 
            data.loc[latest_idx, "relative_volume"] > self.volume_multiplier
        )
        
        # Print current values for debugging
        current_price = data.loc[latest_idx, "close"]
        current_rsi = data.loc[latest_idx, "rsi"]
        current_vol = data.loc[latest_idx, "relative_volume"] if not np.isnan(data.loc[latest_idx, "relative_volume"]) else None
        
        logger.info(f"Current Price: ${current_price:.2f}, RSI: {current_rsi:.2f}, " + 
                    f"Relative Volume: {current_vol:.2f}x" if current_vol else "N/A")
        
        # Check if all conditions are met
        if is_oversold and is_rsi_turning_up and is_bullish and is_high_volume:
            return {
                "timestamp": data.iloc[-1]["timestamp"],
                "price": data.iloc[-1]["close"],
                "rsi": data.iloc[-1]["rsi"],
                "prev_rsi": data.iloc[-2]["rsi"],
                "relative_volume": data.iloc[-1]["relative_volume"],
                "is_optimal_hour": self.is_optimal_trading_hour()
            }
        
        return None
    
    def format_alert_message(self, signal: Dict[str, Any]) -> str:
        """
        Format the alert message for a trading signal.
        
        Args:
            signal: Dictionary with signal details
            
        Returns:
            Formatted alert message
        """
        timestamp = signal["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Calculate target and stop loss prices
        entry_price = signal["price"]
        target_price = entry_price * 1.05  # 5% target
        stop_loss_price = entry_price * 0.98  # 2% stop loss
        
        message = f"""
🚨 RSI OVERSOLD BOUNCE SIGNAL DETECTED 🚨
Time: {timestamp}
Current Price: ${entry_price:.2f}
RSI: {signal['rsi']:.2f} (previous: {signal['prev_rsi']:.2f})
Relative Volume: {signal['relative_volume']:.2f}x average

🎯 TRADE SETUP:
Entry Price: ${entry_price:.2f}
Target Price (5%): ${target_price:.2f}
Stop Loss (2%): ${stop_loss_price:.2f}
Max Hold Time: 4 hours

{'✅ OPTIMAL TRADING HOUR!' if signal["is_optimal_hour"] else '⚠️ Note: Not in optimal trading window'}
        """
        return message
    
    def send_alert(self, message: str) -> None:
        """
        Send an alert with the given message.
        
        Args:
            message: Alert message to send
        """
        # Print to console
        print("\n" + message)
        
        # Log the alert
        logger.info("Trading signal detected!")
        
        # Record the alert time
        self.last_alert_time = datetime.now()
        
        # Write to a file
        with open("btc_trading_signals.txt", "a") as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(message)
            f.write("\n" + "="*50 + "\n")
        
        # Send Telegram alert
        self.send_telegram_alert(message)
    
    def send_telegram_alert(self, message: str) -> None:
        """
        Send a Telegram alert with the given message.
        
        Args:
            message: Alert message to send
        """
        try:
            import requests
            
            # Your Telegram credentials
            BOT_TOKEN = "7931873941:AAGeGIjrieQHIuzO3uHSR6IQTqBV0Osfx20"
            CHAT_ID = "2028552668"
            
            # Format message for Telegram
            # Replace $ with \$ to avoid Markdown parsing issues
            formatted_message = message.replace("$", "\\$")
            
            # Telegram API endpoint for sending messages
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            
            # Parameters for the request
            params = {
                "chat_id": CHAT_ID,
                "text": formatted_message,
                "parse_mode": "MarkdownV2"  # Use Markdown formatting
            }
            
            # Send the message
            response = requests.post(url, params=params)
            
            # Check if the message was sent successfully
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
            else:
                logger.error(f"Failed to send Telegram alert: {response.status_code}, {response.text}")
                
                # Try again without Markdown if there was an error
                params["parse_mode"] = ""
                params["text"] = message  # Use original message
                response = requests.post(url, params=params)
                
                if response.status_code == 200:
                    logger.info("Telegram alert sent successfully (without formatting)")
                else:
                    logger.error(f"Second attempt failed: {response.status_code}, {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")
            
            # Try an alternative method if the import fails
            try:
                import urllib.request
                import urllib.parse
                
                text = urllib.parse.quote_plus(message)
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={text}"
                
                with urllib.request.urlopen(url) as response:
                    logger.info("Telegram alert sent using fallback method")
            except Exception as e2:
                logger.error(f"Fallback method also failed: {str(e2)}")
    
    def send_email_alert(self, message: str) -> None:
        """
        Send an email alert with the given message.
        Requires email configuration - add your SMTP server details.
        
        Args:
            message: Alert message to send
        """
        # Implement email sending logic
        # Example using smtplib:
        '''
        import smtplib
        from email.message import EmailMessage
        
        # Configure these variables with your email details
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        EMAIL_ADDRESS = "your-email@gmail.com"
        EMAIL_PASSWORD = "your-app-password"  # Use app password for Gmail
        RECIPIENT_EMAIL = "your-email@gmail.com"
        
        # Create and send email
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = "BTC Trading Signal Alert"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
        '''
        pass
    
    def run(self) -> None:
        """
        Run the bot continuously, checking for trading signals.
        """
        logger.info("Starting Bitcoin RSI Oversold Bounce Trading Bot with Binance API...")
        logger.info(f"Optimal trading hours (UTC): {self.optimal_hours}")
        
        while True:
            try:
                # Fetch latest data from Binance
                self.price_data = self.fetch_binance_data()
                
                if self.price_data.empty:
                    logger.warning("No data received from Binance, retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Check if we're in an optimal trading hour
                is_optimal_hour = self.is_optimal_trading_hour()
                current_hour = datetime.now(timezone.utc).hour
                
                if is_optimal_hour:
                    logger.info(f"Current hour ({current_hour} UTC) is an optimal trading window.")
                
                # Only proceed with signal checks during optimal hours if enabled
                if not self.prioritize_time_windows or is_optimal_hour:
                    # Check for entry conditions
                    signal = self.check_entry_conditions(self.price_data)
                    
                    # If signal detected, send alert
                    if signal:
                        # Avoid duplicate alerts within a short time
                        if (self.last_alert_time is None or 
                            (datetime.now() - self.last_alert_time).total_seconds() > 1800):  # 30 minutes
                            alert_message = self.format_alert_message(signal)
                            self.send_alert(alert_message)
                
                # Sleep before next check
                logger.info(f"Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.info("Waiting 60 seconds before retry...")
                time.sleep(60)

# Configuration
bot_config = {
    "rsi_period": 14,
    "rsi_threshold": 30.0,  # Consider lowering to 25 for stronger signals
    "volume_multiplier": 1.2,
    "check_interval": 60,  # Check every minute
    "optimal_hours": [1, 7, 5, 16, 13],  # Best trading hours from our analysis
    "prioritize_time_windows": True  # Only alert during optimal hours
}

if __name__ == "__main__":
    # Create and run the bot
    bot = BTCRsiOversoldBot(**bot_config)
    bot.run()