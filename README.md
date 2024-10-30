# Analyzing and Predicting Bitcoin's Price
![image](bitcoin.jpeg)

## 1. Goals of the project
📈 For my graduation project, I have chosen a topic that has fascinated me for quite some time: the history of Bitcoin and its price prediction. The first time I heard about Bitcoin was in a short lecture related to E-commerce at the university lately in 2012. Over 10 years since then, recently intrigued by financial markets and cutting-edge technology, I find cryptocurrencies, particularly Bitcoin, to be a compelling subject due to their disruptive potential and ever-evolving nature.

₿ Bitcoin, since its inception in 2009, has experienced a remarkable journey filled with rapid growth, significant volatility, and major milestones. Its decentralized nature and use of blockchain technology have not only sparked a new wave of digital finance but have also reshaped discussions around traditional monetary systems. However, predicting Bitcoin's price remains a challenging task due to its high volatility and sensitivity to various factors, including regulatory news, market sentiment, and technological developments.

🏅 In this project, I aim to explore the historical trends of Bitcoin's price movements while applying predictive modeling techniques to forecast future prices. By combining a thorough historical analysis with data-driven predictive methods, I hope to shed light on the factors that influence Bitcoin's value and contribute to the broader understanding of cryptocurrency markets. This project will not only enable me to deepen my knowledge of financial modeling and data analysis but will also satisfy my curiosity about Bitcoin and its opportunity in the future, which can also refer for other cryptocurrencies.

Notes: This is an academic project and should not be used in trading or used as a strategy in trading.

## 2. Data source
Sourced from Investing.com (Link: https://www.investing.com/crypto/bitcoin/historical-data)

## 3. Data overview
This dataset, sourced from Investing.com, provides a detailed record of Bitcoin's daily price movements. Capturing key metrics such as opening, closing, high, low prices, and trading volume (currency in USD), the dataset spans from October 10, 2012, to October 10, 2024.

The primary purpose of this dataset is to explore the relationships between these factors and the Bitcoin's price, which can be useful for predictive modelling and understanding how the Bitcoin's price changes.

The dataset is shared in this link: https://docs.google.com/spreadsheets/d/1s_NAKzms-k09NV4E89uJGCJyIpEfxKCg1nCnzQqying/edit?usp=sharing

Columns:

- Date: Date of the recorded data
- Close: Closing price of Bitcoin on the given date (in the original data in Investing website, this column named as Price)
- Open: Opening price of Bitcoin on the given date
- High: Highest price of Bitcoin on the given date
- Low: Lowest price of Bitcoin on the given date
- Volume: Trading volume of Bitcoin on the given date
- Change %: Percentage change in Bitcoin's close price from the previous day

      price_data.sample(10)

![image](https://github.com/user-attachments/assets/4d8cd7c5-7d80-4a22-9072-10b3d129a59a)



## 4. Tools and technologies
For a project focused on Bitcoin price analysis and prediction, Python offers a robust ecosystem with specialized libraries that facilitate data gathering, processing, and modeling. Here are tools and technologies used in the project:

1. **Data Collection and Processing**  
   - **Pandas**: This library is crucial for data manipulation and analysis. It allows users to import, clean, and organize time-series data on Bitcoin prices, often gathered from APIs or CSV files. Pandas is commonly used to resample data, create rolling averages, and handle missing values.
   - **NumPy**: For numerical computations, NumPy provides efficient handling of arrays and mathematical functions, often speeding up operations that involve large datasets.

2. **Data Visualization**  
   - **Matplotlib and Seaborn**: These libraries are essential for visualizing trends, volatility, and other key metrics. Matplotlib offers extensive customization for plotting, while Seaborn simplifies the creation of visually appealing charts, making it easier to spot patterns or anomalies in the data.
   - **Plotly**: Known for its interactive capabilities, Plotly is useful for creating dynamic charts that help in visualizing price fluctuations and user interactions over time, such as candlestick charts often used in financial analysis.

3. **Machine Learning Models**  
   - **Scikit-Learn**: This library provides a comprehensive set of tools for data preprocessing and a wide range of machine learning algorithms, including regression models, support vector machines (SVM), and ensemble methods. Scikit-Learn is valuable for predictive modeling and evaluation through metrics like mean squared error (MSE) and R-squared scores.
   - **TensorFlow and Keras**: For more complex, deep learning-based price prediction models, TensorFlow and its high-level API, Keras, offer neural network architectures such as LSTM (Long Short-Term Memory) networks, which are well-suited for sequential data like Bitcoin prices.

4. **Statistical Analysis and Forecasting**  
   - **Statsmodels**: This library provides statistical models, including ARIMA (AutoRegressive Integrated Moving Average) and GARCH (Generalized Autoregressive Conditional Heteroskedasticity), which are useful for time series forecasting in Bitcoin’s volatile price environment. Statsmodels also enables hypothesis testing and other statistical analysis to evaluate the significance of results.
  
## 5. Key insights

   

## 6. Hypotheses

## 7. Recommendations





