import pandas as pd
import os
import torch
import streamlit as st
from model_predictor import predict_stock
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests


os.environ["PYTHONWATCHER_IGNORE_MODULES"] = "torch"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(
    page_title="TradeSense AI",
    page_icon='assets/logo.png' if os.path.exists('assets/logo.png') else None,
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stSidebar {
        background-color: #2D2D2D;
    }
    .stButton>button {
        color: #E0E0E0;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stSelectbox {
        color: #E0E0E0;
    }
    h1, h2, h3 {
        color: #E0E0E0;
    }
    .stDataFrame {
        color: #E0E0E0;
    }
    .stPlotlyChart {
        background-color: #2D2D2D;
    }
    .stMetric {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .stMetric .metric-value {
        color: #4CAF50;
    }
    .stWarning {
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

def get_current_day_news(stock_name):
    api_key = "ebbfdb3571835060514db9497443b4f3"
    # yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    base_url = "http://api.mediastack.com/v1/news"
    params = {
        "access_key": api_key,
        "keywords": stock_name,
        "languages": "en",
        # "date": yesterday_date, 
        "limit": 5  
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'data' in data:
            return data['data']  
        else:
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []


@st.cache_data
def load_data(stock_name):
    return pd.read_csv(os.path.join('data', f'merged_{stock_name.lower().replace(" ", "")}_data_with_sentiment.csv'))

company_info = {
    'Reliance': {
        'Info':('Reliance Industries Limited (RIL), based in Mumbai, India, is a major conglomerate founded by Dhirubhai Ambani in 1960. It operates across sectors like petrochemicals, refining, telecommunications, and retail, making it one of India’s largest companies by market capitalization. Notably, RIL revolutionized the telecom industry with Reliance Jio, offering affordable data services, and is also focused on sustainability and green energy initiatives.'),
        'Market Cap': '₹ 17,47,896 Cr',
        'Current Price': '₹ 1,292',
        'P/E ratio': '25.76',
        'Dividend yield': '0.39%',
        'EPS': '50.19',
        'ROE': '10.48%',
        'Debt to equity': '0.25'     
    },
    'Tata Motors': {
        'Info': ('Tata Motors, part of the Tata Group, is a leading Indian automotive manufacturer headquartered in Mumbai. Founded in 1945, it produces a wide range of vehicles, including cars, trucks, buses, and electric vehicles (EVs). Tata Motors is known for its innovation in the automotive sector and has expanded globally through acquisitions like Jaguar Land Rover. The company focuses on sustainable mobility and has been increasing its presence in the EV market while maintaining a strong foothold in commercial vehicles.'),
        'Market Cap': '₹2,89,216 Cr',
        'Current Price': '₹787.05',
        'P/E ratio': '8.67',
        'Dividend yield': '0.77%',
        'EPS': '90.58',
        'ROE': '48.9%',
        'Debt to equity': '0.08' 
        
    },
     'Airtel': {
         'Info': ('Bharti Airtel, headquartered in New Delhi, is a leading global telecommunications company founded in 1995. It operates in multiple countries, primarily across Asia and Africa, offering services like mobile, broadband, and digital TV. Airtel is one of India’s largest telecom providers and played a pivotal role in making mobile and internet services accessible to millions. Known for its innovations in 4G and 5G networks, Bharti Airtel is also expanding into digital services, fintech, and enterprise solutions.'),
        'Market Cap': '₹9,33,754.20 Cr',
        'Current Price': '₹1,577.65',
        'P/E ratio': '72.44',
        'Dividend yield': '0.49%',
        'EPS': '₹21.54',
        'ROE': '16.32%',
        'Debt to equity': '2.59' 
    },
    'Maruti Suzuki': {
        'Info': ("Maruti Suzuki, headquartered in New Delhi, is India's largest automobile manufacturer and a subsidiary of Suzuki Motor Corporation, Japan. Founded in 1981, it dominates the Indian car market with a wide range of affordable, fuel-efficient cars. Known for popular models like the Alto and Swift, Maruti Suzuki has been a key player in shaping India's automotive landscape."),
        'Market Cap': '₹3,44,763 Cr',
        'Current Price': '₹750',
        'P/E ratio': '24.55',
        'Dividend yield': '1.14%',
        'EPS': '₹445.97',
        'ROE': '16.84%',
        'Debt to equity': '0.00' 
    },
     'Infosys': {
        'Info': ('Infosys, based in Bengaluru, is a global leader in consulting, technology, and next-generation digital services. Founded in 1981, it offers services like IT consulting, business process management, and software development. Infosys is renowned for its innovation and has a significant presence worldwide, driving digital transformation for enterprises.'),
        'Market Cap': '₹7,71,259 Cr',
        'Current Price': '₹3400',
        'P/E ratio': '28.62',
        'Dividend yield': '2.48%',
        'EPS': '64.90',
        'ROE': '32.46%',
        'Debt to equity': '0.0'
    },
    'Asian Paints': {
        'Info': ('Asian Paints, headquartered in Mumbai, is India’s largest paint company and a major player in the global coatings industry. Established in 1942, it offers a wide range of products, including decorative paints and industrial coatings. The company is known for its innovation in color solutions and extensive retail network.'),
        'Market Cap': '₹2,38,255 Cr',
        'Current Price': '₹ 2,480',
        'P/E ratio': '52.15',
        'Dividend yield': '1.34%',
        'EPS': '48',
        'ROE': '32.08%',
        'Debt to equity': '0'
    },
    'JSW': {
        'Info': ('JSW Steel, part of the JSW Group, is one of India’s largest steel producers, headquartered in Mumbai. Established in 1982, the company is known for its advanced technology and large-scale operations in the production of steel products used across industries like construction, automotive, and infrastructure'),
        'Market Cap': '₹2,33,186 Cr',
        'Current Price': '₹921.15',
        'P/E ratio': '46.66',
        'Debt to equity': '0.71',
        'EPS':'33.16',
        'ROE': '12.6%',
        'Dividend yield': '0.77%'
    },
    'Mahindra': {
        'Info': ('Mahindra & Mahindra, based in Mumbai, is a flagship company of the Mahindra Group and one of India’s largest automakers. Founded in 1945, it is known for manufacturing SUVs, commercial vehicles, and tractors. The company is also expanding into electric vehicles and aims to be a leader in sustainable mobility.'),
        'Market Cap': '₹3,60,605 Cr',
        'Current Price': '₹2966.10',
        'P/E ratio': '30.39',
        'Dividend yield': '0.73%',
        'EPS': '88',
        'ROE': '22.55%',
        'Debt to equity': '0.02'
    },
    'Hyundai': {
        'Info': ('Hyundai Motor India, a subsidiary of the South Korean Hyundai Motor Company, is one of the leading automobile manufacturers in India. Established in 1996, Hyundai is known for its modern, feature-rich vehicles like the i20 and Creta, and has been a key player in the Indian car market, offering a wide range of models.'),
        'Market Cap': '₹1,54,566 Cr',
        'Current Price':'1916.55',
        'P/E ratio': '25.51',
        'Dividend yield': '6.98%',
        'EPS': '74.58',
        'ROE': '39.45%',
        'Debt to equity': '0.07'
    },
    'Bajaj': {
        'Info': (' Bajaj Auto, headquartered in Pune, is a leading Indian manufacturer of motorcycles, scooters, and three-wheelers. Founded in 1945, it is known for popular models like the Pulsar and Chetak, and has a strong presence in the two-wheeler segment both in India and globally. Bajaj is also expanding its focus on electric vehicles.'),
        'Market Cap': '₹2,52,203.13 Cr',
        'Current Price': '₹6509.40',
        'P/E ratio':'34.28',
        'EPS': '₹248.39',
        'Dividend yield': '0.55%',
        'Debt to equity': 'Data unavailable currently',
        'ROE': 'Data unavailable currently'
    }
}

def app():
    st.title('TradeSense AI')
    st.markdown("""
        Welcome to the Stock Prediction App! Here you can predict stock trends based on the GRU (Gated Recurrent Unit) model. 
        You can get predictions for various stocks, explore detailed stock data, and see today's news related to your selected stock.
    """)

    st.sidebar.header('TradeSense AI')
    if os.path.exists('assets/logo.png'):
        st.sidebar.image('assets/logo.png')

    stock_options = ['Reliance', 'Tata Motors', 'JSW', 'Airtel', 'Infosys', 'Asian Paints', 
                     'Hyundai', 'Bajaj', 'Maruti Suzuki', 'Mahindra']
    stock_name = st.sidebar.selectbox('Select Stock', stock_options)

    try:
        stock_data = load_data(stock_name)
    except FileNotFoundError:
        st.warning(f"⚠️ No data available for {stock_name}. Please check the data file.")
        return

    page_selection = st.sidebar.radio("📈 Select a Page", ["Home", "Stocks", "Predictions"])
    
    if page_selection == "Home":
            st.header(f"{stock_name} - Stock Prediction")
            
            st.subheader("📰 Today's News:")
            news_items = get_current_day_news(stock_name)
            if news_items:
                for news in news_items:
                    st.markdown(f'<a href="{news["url"]}" target="_blank" style="color:#FC1D93; font-size:18px; font-weight:bold;">{news["title"]}</a>', unsafe_allow_html=True)
                    st.write("")
            else:
                st.write("No news articles found for this stock.")
                
            st.markdown("#### 📈 Stock Recommendation")
            if st.button(f'Get Prediction for {stock_name}'):
                st.write(f"Running prediction for {stock_name}...")
                try:
                    recommendation = predict_stock(stock_name, stock_data, return_recommendation=True)
                    st.success(f"Recommendation for {stock_name}: {recommendation}")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        
    elif page_selection == "Stocks":
        show_stocks_page(stock_name, stock_data)

    elif page_selection == "Predictions":
        show_predictions_page(stock_name)


def show_stocks_page(stock_name, stock_data):
    st.title(f"{stock_name} Stock Data Overview")
    st.markdown("""
        This page provides additional data on the selected stock. 
        You can explore trends, technical indicators, and key performance metrics to make informed decisions.
    """)
    stock_details = company_info.get(stock_name, {})
    market_cap = stock_details.get('Market Cap', "Data unavailable")
    pe_ratio = stock_details.get('P/E ratio', "Data unavailable")
    dividend_yield = stock_details.get('Dividend yield', "Data unavailable")
    eps = stock_details.get('EPS', "Data unavailable")
    roe = stock_details.get('ROE', "Data unavailable")
    debt_to_equity = stock_details.get('Debt to equity', "Data unavailable")

    st.subheader("Stock Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Cap", market_cap)
    col2.metric("P/E Ratio", pe_ratio)
    col3.metric("Dividend Yield", dividend_yield)
    col4.metric("EPS", eps)

    st.subheader("Stock Price Analysis")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data['date'],
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Price'))
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['Close'].rolling(window=20).mean(), name='20-day MA', line=dict(color='#FFA500')))
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['Close'].rolling(window=50).mean(), name='50-day MA', line=dict(color='#FF4136')))
    fig.update_layout(
        title=f"{stock_name} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='#2D2D2D',
        paper_bgcolor='#2D2D2D',
        font=dict(color='#E0E0E0')
    )
    st.plotly_chart(fig)

    st.subheader("Trading Volume Analysis")
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=stock_data['date'], y=stock_data['Volume'], name='Volume', marker_color='#4CAF50'))
    fig_volume.update_layout(
        title=f"{stock_name} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        plot_bgcolor='#2D2D2D',
        paper_bgcolor='#2D2D2D',
        font=dict(color='#E0E0E0')
    )
    st.plotly_chart(fig_volume)

    st.subheader("Additional Financial Metrics")
    col1, col2 = st.columns(2)
    col1.metric("ROE", roe)
    col2.metric("Debt to Equity", debt_to_equity)


def get_stock_prediction(stock_name):
    try:
        stock_data = load_data(stock_name)
        predicted_high, predicted_low = predict_stock(stock_name, stock_data)
        recommendation = predict_stock(stock_name, stock_data) 
        return predicted_high, predicted_low, recommendation
    except Exception as e:
        return None, None, f"Error: {str(e)}"
    
def graph(stock_name):
    df = load_data(stock_name)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    yearly_avg = df.groupby('year')[['High', 'Low']].mean().reset_index()
    st.write(yearly_avg)
    years = yearly_avg['year'].astype(str).tolist() 
    avg_predicted_high = yearly_avg['High'].tolist()
    avg_predicted_low = yearly_avg['Low'].tolist()
    return years, avg_predicted_high, avg_predicted_low

def show_predictions_page(stock_name):
    st.title("🔮 Stock Predictions and Forecasts")
    st.markdown("""
    <style>
    .header {
        font-size:36px;
        font-weight:bold;
        text-align:center;
        color: #4CAF50;
    }
    .subheader {
        font-size:24px;
        font-weight:bold;
        color: #4CAF50;
        margin-top:20px;
    }
    </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Predicted Prices"])

    with tabs[0]:
        if stock_name:
            st.markdown(f"**Selected Stock for Prediction:** {stock_name}")
            predicted_high, predicted_low, recommendation = get_stock_prediction(stock_name)

            st.subheader("Predicted High and Low Prices for next day")
            st.markdown(f"- **Predicted High:** ₹{predicted_high:.2f}")
            st.markdown(f"- **Predicted Low:** ₹{predicted_low:.2f}")

            st.markdown('<div class="subheader">📈 Predicted High and Low Prices</div>', unsafe_allow_html=True)
            
            if stock_name: 
                years, avg_predicted_high, avg_predicted_low = graph(stock_name)
                data = pd.DataFrame({
                    'Year': years,
                    'Predicted High': avg_predicted_high,
                    'Predicted Low': avg_predicted_low
                })
                fig, ax = plt.subplots(figsize=(2.5, 2.5))  

                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                ax.fill_between(years, avg_predicted_low, avg_predicted_high, color='cyan', alpha=0.2, label='Predicted Range')
                sns.lineplot(x=years, y=avg_predicted_high, marker='o', label='Predicted High', color='cyan', linewidth=1.2, ax=ax)
                sns.lineplot(x=years, y=avg_predicted_low, marker='o', label='Predicted Low', color='red', linewidth=1.2, ax=ax)

                ax.set_title("Predicted High and Low Prices Over the Years", fontsize=7, color='white', pad=10)
                ax.set_xlabel("Year", fontsize=5, color='white', labelpad=5)
                ax.set_ylabel("Price (₹)", fontsize=5, color='white', labelpad=5)

                ax.tick_params(axis='both', colors='white', labelsize=5)

                ax.grid(False) 

                ax.legend(loc='upper left', fontsize=5, frameon=False, labelcolor='white')

                st.pyplot(fig)


    st.markdown("""
        <hr>
        <footer style="text-align:center; color:gray;">
            Powered by <b>Deep Learning & Streamlit</b> | Designed for smarter stock decisions.
        </footer>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    app()
