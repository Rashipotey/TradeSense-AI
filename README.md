# **TradeSense AI**  
### **AI-Powered Stock Market Analysis and Prediction System**  

TradeSense AI is an advanced stock market prediction platform that leverages deep learning, financial indicators, and sentiment analysis to provide actionable investment insights.  

## **Features**  
✅ **Buy/Hold/Sell Recommendations** – AI-driven decision-making based on real-time and historical data.  
✅ **Sentimental Analysis** – Extracts insights from news and social media to assess market sentiment.  
✅ **Technical Indicators** – Incorporates SMA, EMA, RSI, Bollinger Bands, MACD, and volume analysis.  
✅ **Trend Forecasting** – Predicts stock movements using GRU-based deep learning models.  
✅ **Performance Metrics** – Evaluates model accuracy with MAE, RMSE, and feature importance breakdown.  
✅ **Interactive UI** – User-friendly Streamlit dashboard for seamless stock analysis.  

## **Installation**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/tradesense-ai.git
cd tradesense-ai
```
### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
### **3️⃣ Set Up API Keys**  
Create a `.env` file and add your API credentials:  
```
STOCK_DATA_API_KEY="your_api_key"
NEWS_API_KEY="your_api_key"
```
### **4️⃣ Run the Application**  
```bash
streamlit run app.py
```

## **Stock Analysis Features**  
- **Home Page** – Select a stock to view AI-powered recommendations, news sentiment, pros & cons, and company details.  
- **Stocks Page** – Get a detailed view of stock trends, moving averages, and market cap insights.  
- **Predictions Page** – Visualize trend forecasts, compare model performance, and explore key influencing factors.  

## **Tech Stack**  
- **Frontend**: Streamlit  
- **Backend**: Python, TensorFlow/Keras  
- **Data Sources**: Stock APIs, News APIs  
- **ML Models**: GRU, Sentiment Analysis (DistilBERT)  
- **Deployment**: Streamlit Community Cloud  
