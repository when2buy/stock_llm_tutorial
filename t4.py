'''
Advanced Stock Analysis Dashboard with:
- Multiple analysis strategies
- Technical indicators
- Historical comparisons
- Portfolio tracking
- Real-time updates
'''

import asyncio
from datetime import datetime, timedelta

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from openai import AsyncOpenAI
from plotly.subplots import make_subplots

# LLM Configuration
conf = {
    "openai": (
        "https://api.openai.com/v1",
        # Replace with your own API key
        "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",
        "chatgpt-4o-latest"
    )
}

# Initialize LLM clients
clients = {}
for name, (base_url, api_key, _) in conf.items():
    clients[name] = AsyncOpenAI(base_url=base_url, api_key=api_key)

class StockAnalyzer:
    def __init__(self):
        self.cached_data = {}
        
    async def get_stock_data(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch and cache stock data"""
        cache_key = f"{ticker}_{period}"
        if cache_key in self.cached_data:
            if datetime.now() - self.cached_data[cache_key]['timestamp'] < timedelta(minutes=5):
                return self.cached_data[cache_key]['data']
        
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        self.cached_data[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        return data

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df

    def create_analysis_chart(self, data: pd.DataFrame, ticker: str) -> go.Figure:
        """Create an advanced analysis chart with technical indicators"""
        df = self.calculate_technical_indicators(data)
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.6, 0.2, 0.2])

        # Candlestick chart with MA
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='MA20',
            line=dict(color='orange')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            name='MA50',
            line=dict(color='blue')
        ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume'
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI'
        ), row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f"{ticker} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,
            xaxis_rangeslider_visible=False
        )

        return fig

async def get_llm_analysis(data: pd.DataFrame, ticker: str, strategy: str):
    """Get comprehensive LLM analysis"""
    if "openai" not in clients:
        raise ValueError("OpenAI client not configured")
    
    client = clients["openai"]
    _, _, model = conf["openai"]
    
    # Calculate additional metrics
    returns = data['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    prompt = f"""
    Provide a comprehensive analysis for {ticker} using {strategy} approach:
    
    Latest Data:
    {data.tail().to_string()}
    
    Key Metrics:
    - Current Price: ${data['Close'].iloc[-1]:.2f}
    - Price Change (Period): ${(data['Close'].iloc[-1] - data['Close'].iloc[0]):.2f}
    - Daily Volume (Avg): {data['Volume'].mean():,.0f}
    - Volatility (Annual): {volatility:.2%}
    - Trading Range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}
    
    Please provide:
    1. Technical Analysis (including support/resistance levels)
    2. Market Trends and Pattern Recognition
    3. Volume Analysis and Significance
    4. Risk Assessment
    5. Short-term and Long-term Outlook
    6. Trading Recommendations with Entry/Exit Points
    """
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert stock market analyst with deep knowledge of technical and fundamental analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

async def process_analysis(ticker: str, strategy: str, period: str):
    """Main analysis process"""
    try:
        analyzer = StockAnalyzer()
        
        # Fetch stock data
        stock_data = await analyzer.get_stock_data(ticker, period)
        
        # Create advanced visualization
        stock_chart = analyzer.create_analysis_chart(stock_data, ticker)
        
        # Get AI analysis
        analysis = await get_llm_analysis(stock_data, ticker, strategy)
        
        return stock_chart, analysis
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        return None, error_msg

# Gradio Interface
with gr.Blocks(title="Advanced Stock Analysis Dashboard") as demo:
    gr.Markdown("""
    # 📈 Advanced Stock Analysis Dashboard
    Analyze stocks with technical indicators and AI-powered insights
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            stock_code = gr.Textbox(
                label="Stock Symbol",
                value="AAPL",
                placeholder="Enter stock symbol (e.g., AAPL)"
            )
            
            with gr.Row():
                strategy_type = gr.Dropdown(
                    label="Strategy Type",
                    choices=[
                        "Predefined Strategy",
                        "Custom Strategy"
                    ],
                    value="Predefined Strategy"
                )
            
            with gr.Row():
                predefined_strategy = gr.Dropdown(
                    label="Predefined Strategies",
                    choices=[
                        "Technical Analysis",
                        "Trend Following",
                        "Momentum Trading",
                        "Mean Reversion",
                        "Volume Analysis"
                    ],
                    value="Technical Analysis",
                    visible=True
                )
                custom_strategy = gr.Textbox(
                    label="Custom Strategy",
                    placeholder="Enter your custom analysis strategy",
                    visible=False
                )
            
            period = gr.Dropdown(
                label="Time Period",
                choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                value="3mo"
            )
            analyze_btn = gr.Button("🔍 Analyze Stock", variant="primary")
        
        with gr.Column(scale=2):
            chart_output = gr.Plot(label="Technical Analysis Chart")
            analysis_report = gr.Textbox(
                label="AI Analysis Report",
                lines=15,
                interactive=False
            )
    
    def update_strategy_visibility(strategy_type):
        return {
            predefined_strategy: gr.update(visible=strategy_type == "Predefined Strategy"),
            custom_strategy: gr.update(visible=strategy_type == "Custom Strategy")
        }
    
    strategy_type.change(
        fn=update_strategy_visibility,
        inputs=[strategy_type],
        outputs=[predefined_strategy, custom_strategy]
    )
    
    def get_strategy(strategy_type, predefined, custom):
        return custom if strategy_type == "Custom Strategy" else predefined
    
    async def handle_analysis(ticker, strategy_type, predefined, custom, period):
        strategy = get_strategy(strategy_type, predefined, custom)
        return await process_analysis(ticker, strategy, period)
    
    analyze_btn.click(
        fn=handle_analysis,
        inputs=[stock_code, strategy_type, predefined_strategy, custom_strategy, period],
        outputs=[chart_output, analysis_report]
    )

if __name__ == "__main__":
    demo.launch(share=True)
