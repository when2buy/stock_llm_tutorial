'''
Use gradio to visualize the stock data
'''

from datetime import datetime

import gradio as gr
import plotly.graph_objects as go
import yfinance as yf


def create_stock_chart(ticker: str):
    # Fetch stock data using yfinance
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")  # Get 1 month of data
    
    # generate stock chart using plotly
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close']
            )
        ]
    )
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        yaxis_title="Price",
        xaxis_title="Date"
    )
    
    return fig

async def process_analysis(ticker: str, strategy: str):
    try:
        stock_chart = create_stock_chart(ticker)
        dummy_text = "This is a dummy text\n"
        dummy_text += f"Analysis {ticker} with strategy {strategy}"
        return stock_chart, dummy_text
    except Exception as e:
        return None, f"Error: {str(e)}"


with gr.Blocks() as demo:
    stock_code = gr.Textbox(label="Stock Code", value="AAPL")
    strategy = gr.Textbox(label="Strategy", value="Technical Analysis")
    chart_output = gr.Plot()
    analysis_report = gr.Textbox(label="Analysis Report", lines=5)

    btn = gr.Button("Analyze")
    btn.click(process_analysis, 
             inputs=[stock_code, strategy], 
             outputs=[chart_output, analysis_report])

demo.launch()