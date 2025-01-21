# Show the stock data
# and use llm to analyze the stock data
import yfinance as yf

stock = yf.Ticker("AAPL")
data = stock.history(period="5d")
print(data)

# use llm to analyze the stock data
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze the stock data: " + data.to_string()}]
)
print(response.choices[0].message.content)