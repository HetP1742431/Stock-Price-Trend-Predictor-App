# 📈 Stock Price Trend Predictor App (ML)

The Stock Price Trend Predictor App is a machine learning project designed to forecast stock price trends using historical data. Leveraging Long Short-Term Memory (LSTM) networks, the app provides users with insightful predictions and visualizations of stock performance.

## 🛠 Core Features
- **Data Collection & Preprocessing**: 
  - The app fetches historical stock data using the `yfinance` API, ensuring up-to-date and reliable financial information.
  - Extensive data cleaning and manipulation are performed using `Pandas` and `NumPy`, including handling missing values, calculating moving averages, and normalizing the data for model input.

- **LSTM Model for Prediction**:
  - A robust LSTM model built with `TensorFlow` and `Keras` is used to predict stock price trends. The model is trained on 10 years of stock data, providing high accuracy with an average MAPE of less than 5%.
  - Data is scaled using `MinMaxScaler` from `scikit-learn` to optimize the LSTM network's performance.

- **Visualizations**:
  - Historical stock prices and moving averages are visualized using `Matplotlib`, allowing users to easily understand past performance and trends.
  - The app provides interactive plots that display both historical data and predicted trends, offering a comprehensive view of stock behavior.

- **User Interface**:
  - The model is deployed via `Streamlit`, providing a clean and intuitive user interface where users can input stock ticker symbols and view predictions.
  - The app displays the next day's estimated closing price, offering a practical tool for investors and analysts.

## 🛠 Tech Stack
- **Machine Learning**: TensorFlow, Keras
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib
- **Data Scaling**: scikit-learn
- **Data Collection**: yfinance API
- **Deployment**: Streamlit

## 📋 How to Run the Project Locally

### Prerequisites
- Python 3.6 or higher
- Pip package manager

### Installation and Initialization of the app
1. Clone the repository:
   ```bash
   git clone https://github.com/HetP1742431/Stock-Price-Trend-Predictor-App.git
   cd Stock-Price-Trend-Predictor-App
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open your web browser and go to http://localhost:8501 to view the app.

## Usage:
- Enter a Ticker Symbol: Start by entering the stock ticker symbol (e.g., AAPL, MSFT) in the provided input field.
- View Historical Data: The app will display the historical data and various financial indicators like moving averages.
- Predict Future Prices: The LSTM model predicts the next day's closing price, which is displayed along with historical trends.

## 💡 What I Learned
- Gained deep expertise in developing and optimizing LSTM networks for time series forecasting, particularly in the context of financial data.
- Enhanced skills in data preprocessing, including cleaning, normalization, and feature engineering, using powerful libraries like Pandas and NumPy.
- Developed proficiency in using TensorFlow and Keras for building and training deep learning models, and in evaluating model performance with metrics like MAPE.
- Improved problem-solving abilities through troubleshooting and fine-tuning the model, ensuring accuracy and reliability.
- Learned how to deploy a machine learning model using Streamlit, creating an interactive user interface that makes complex predictions accessible to users.

This project was a valuable experience that not only strengthened my technical skills in machine learning and data science but also enhanced my ability to deploy practical applications. I invite you to explore the project, check out the code, and see how machine learning can be applied to financial forecasting.

- **Live Demo Video**: [Demo video](https://drive.google.com/file/d/1P45UFpg83gZmrZqvToo4TQLin98Fqb2k/view?usp=drive_link).
