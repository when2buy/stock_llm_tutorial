'''
Retrive stock data from yfinance asynchronously
'''

import asyncio
from datetime import datetime

import yfinance as yf


async def analyze_stock(ticker: str):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5d")
    return data

async def process_stock(ticker: str, strategy: str):
    try:
        stock_data = await analyze_stock(ticker)
        # print(stock_data)

        print("Stock data retrieved successfully, now apply strategy: ", strategy)
        report = f'''
        Stock Code: {ticker}
        Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Analysis Data: 
        {stock_data}
        Strategy: {strategy}
        '''
        print("Analysis report generated successfully:\n", report)
    except Exception as e:
        print("Error retrieving stock data or applying strategy: ", e)

if __name__ == "__main__":
    asyncio.run(process_stock("AAPL", "Tell me the trend of the stock"))