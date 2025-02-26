import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone
import pytz
import json
from typing import List, Dict, Any, Optional
import os

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
    
    def fetch_data_from_coingecko(self, days: int = 1) -> pd.DataFrame:
        """
        Fetch Bitcoin data from CoinGecko API.
        
        Args:
            days: Number of days of historical data to retrieve (default: 1)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # CoinGecko API for Bitcoin market data
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "minute"
            }
            
            logger.info(f"Fetching data from CoinGecko for the past {days} day(s)...")
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Error fetching data: {response.status_code}, {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Process price data
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            
            # Process volume data
            volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
            
            # Merge price and volume data
            merged_data = pd.merge_asof(prices, volumes, on="timestamp")
            
            # Create OHLCV data (CoinGecko only provides close prices and volumes)
            # We'll use the close price for open, high, and low for the current minute
            # For a production system, you'd need a more granular data source
            merged_data["open"] = merged_data["price"].shift(1)
            merged_data["high"] = merged_data["price"]
            merged_data["low"] = merged_data["price"]
            merged_data["close"] = merged_data["price"]
            
            # Drop the first row (has NaN in open) and the price column (redundant)
            merged_data = merged_data.dropna()
            merged_data = merged_data.drop(columns=["price"])
            
            logger.info(f"Successfully fetched {len(merged_data)} data points")
            return merged_data
        
        except Exception as e:
            logger.error(f"Error fetching data from CoinGecko: {str(e)}")
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
        This is a placeholder - implement your preferred alert method here.
        
        Args:
            message: Alert message to send
        """
        # Print to console
        print("\n" + message)
        
        # Log the alert
        logger.info("Trading signal detected!")
        
        # Record the alert time
        self.last_alert_time = datetime.now()
        
        # === ALERT IMPLEMENTATION OPTIONS ===
        
        # Option 1: Write to a file
        with open("btc_trading_signals.txt", "a") as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(message)
            f.write("\n" + "="*50 + "\n")
        
        # Option 2: Send an email (uncomment and configure)
        # self.send_email_alert(message)
        
        # Option 3: Send a Telegram message (uncomment and configure)
        # self.send_telegram_alert(message)
    
    def send_email_alert(self, message: str) -> None:
        """
        Send an email alert with the given message.
        Requires email configuration - add your SMTP server details.
        
        Args:
            message: Alert message to send
        """
        # Implement email sending logic here
        # Example using smtplib (you'd need to import it)
        '''
        import smtplib
        from email.message import EmailMessage
        
        # Configure these variables with your email details
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        EMAIL_ADDRESS = "your-email@gmail.com"
        EMAIL_PASSWORD = "your-app-password"  # Use app password for Gmail
        RECIPIENT_EMAIL = "recipient-email@example.com"
        
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
    
    def send_telegram_alert(self, message: str) -> None:
        """
        Send a Telegram alert with the given message.
        Requires Telegram bot configuration.
        
        Args:
            message: Alert message to send
        """
        # Implement Telegram sending logic here
        # Example using python-telegram-bot (you'd need to install it)
        '''
        import telegram
        
        # Configure these variables with your Telegram details
        BOT_TOKEN = "your-bot-token"
        CHAT_ID = "your-chat-id"
        
        try:
            bot = telegram.Bot(token=BOT_TOKEN)
            bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
            logger.info("Telegram alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")
        '''
        pass
    
    def run(self) -> None:
        """
        Run the bot continuously, checking for trading signals.
        """
        logger.info("Starting Bitcoin RSI Oversold Bounce Trading Bot...")
        logger.info(f"Optimal trading hours (UTC): {self.optimal_hours}")
        
        while True:
            try:
                # Fetch latest data
                self.price_data = self.fetch_data_from_coingecko()
                
                if self.price_data.empty:
                    logger.warning("No data received, retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                # Check if we're in an optimal trading hour
                is_optimal_hour = self.is_optimal_trading_hour()
                
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
                
                # Log current status
                current_time = datetime.now()
                current_hour = current_time.hour
                if is_optimal_hour:
                    logger.info(f"Current hour ({current_hour}) is an optimal trading window.")
                
                # Sleep before next check
                logger.info(f"Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.info("Waiting 120 seconds before retry...")
                time.sleep(120)

# Example configuration - customize these parameters based on your preferences
bot_config = {
    "rsi_period": 14,
    "rsi_threshold": 30.0,
    "volume_multiplier": 1.2,
    "check_interval": 60,  # Check every minute
    "optimal_hours": [1, 7, 5, 16, 13],  # Best trading hours from our analysis
    "prioritize_time_windows": True  # Only alert during optimal hours
}

if __name__ == "__main__":
    # Create and run the bot
    bot = BTCRsiOversoldBot(**bot_config)
    bot.run()