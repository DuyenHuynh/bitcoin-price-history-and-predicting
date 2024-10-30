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

      price_data.info()
![image](https://github.com/user-attachments/assets/f7c5dbe1-5271-443a-a47b-c3728f45763a)


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
### a. Basic EDA insights
- The Open, Close, Low, and High prices appear to follow the same trend throughout the specified period.

        plt.figure(figsize=(16,5))
        plt.plot(price_data['Date'], price_data['Open'], color='blue', label='Open')
        plt.plot(price_data['Date'], price_data['Close'], color='blue', label='Close')
        plt.plot(price_data['Date'], price_data['Low'], color='blue', label='Low')
        plt.plot(price_data['Date'], price_data['High'], color='blue', label='High')

        plt.title('Bitcoin price from 2012 - 2024')
        plt.show()

<img width="657" alt="image" src="https://github.com/user-attachments/assets/67a4c29d-0a0f-4cce-a3b7-2f5bcc7ab517">

<img width="657" alt="image" src="https://github.com/user-attachments/assets/8baf7015-7cda-4663-a362-16ee267a91a7">

- Standard deviation (std) is higher than the mean value, indicating high variability.

        price_data.describe()
<img width="552" alt="image" src="https://github.com/user-attachments/assets/eddde7f0-9e36-4485-a4fa-ff4abc8a46f7">

- The Open, High, Low, and Close prices are left-skewed.

      fig = make_subplots(rows=2, cols=2, subplot_titles=('<b>Distr. of Open Price</b>',
                                                          '<b>Distr. of Close Price</b>',
                                                         '<b>Distr. of Low Price</b>',
                                                         '<b>Distr. of High Price</b>',
                                                         ))
      
      fig.add_trace(go.Histogram(x=price_data['Open'].dropna()), row=1, col=1)
      fig.add_trace(go.Histogram(x=price_data['Close'].dropna()), row=1, col=2)
      fig.add_trace(go.Histogram(x=price_data['Low'].dropna()), row=2, col=1)
      fig.add_trace(go.Histogram(x=price_data['High'].dropna()), row=2, col=2)
      
      # Update visual layout
      fig.update_layout(
          showlegend=False,
          width=600,
          height=400,
          autosize=False,
          margin=dict(t=15, b=0, l=5, r=5),
          template="plotly_white",
          colorway=px.colors.qualitative.Prism ,
      )
      
      # update font size at the axes
      fig.update_coloraxes(colorbar_tickfont_size=10)
      # Update font in the titles: Apparently subplot titles are annotations (Subplot font size is hardcoded to 16pt · Issue #985)
      fig.update_annotations(font_size=12)
      # Reduce opacity
      fig.update_traces(opacity=0.75)
      
      fig.show()
![image](https://github.com/user-attachments/assets/1eb0f052-9bed-425e-a6a8-5e41231d8309)

- The time series of the Bitcoin price is non-stationary.

      def stationarity_check(ts):
          plt.plot(ts, label='Original')
          plt.plot(ts.rolling(window=30, center=False).mean(), label='Rolling Mean')
          plt.plot(ts.rolling(window=30, center=False).std(), label='Rolling Std')
          plt.grid()
          plt.legend()
          plt.show()
      
          adf = adfuller(ts, autolag='AIC')
          padf = pd.Series(adf[:4], index=['T Statistic','P-Value','#Lags Used','#Observations Used'])
          for k,v in adf[4].items():
              padf['Critical value {}'.format(k)]=v
          print(padf)
      
      stationarity_check(price_data['Close'])

<img width="425" alt="image" src="https://github.com/user-attachments/assets/3dc500b7-3282-4a3d-842e-a06442117191">

- ACF & PACF suggest the presence of autocorrelation, seasonality, and trend.

      gp = plot_acf(price_data['Close'])
      plt.show()
<img width="369" alt="image" src="https://github.com/user-attachments/assets/93fd6afe-895e-4764-ac0e-bbc8ac3c5f5c">

      gp = plot_pacf(price_data['Close'], method='ywm')
      plt.show()

<img width="356" alt="image" src="https://github.com/user-attachments/assets/0bec56cd-25bb-4669-8f24-0de1bb79649d">

- The trend in Bitcoin prices is non-linear.

      result_add = sm.tsa.seasonal_decompose(x=price_data['Close'], model='additive', extrapolate_trend='freq', period=365)
      result_add.trend
      
      frame = {
          'price_date': price_data['Date'],
          'observed': result_add.observed,
          'trend': result_add.trend,
          'seasonal': result_add.seasonal,
          'residuals': result_add.resid
      }
      
      decomposed_ts_df = pd.DataFrame(frame).reset_index()

<img width="589" alt="image" src="https://github.com/user-attachments/assets/9a47b991-39cd-4a27-85e1-a25a8d5f7895">

- Change points highlight 'black swan' effects on Bitcoin prices that occur on or near these dates.

      def plot_change_points_ruptures(price_data, ts, ts_change_all, title):
      
          plt.figure(figsize=(16,4))
          plt.plot(price_data.index, ts)
          for x in [price_data.iloc[idx-1].name for idx in ts_change_all]:
              plt.axvline(x, lw=2, color='red')
      
          plt.title(title)
          plt.show()
      
      data = price_data[['Date', 'Close']]
      tsd = np.array(data['Close'])
      
      detector = rpt.Pelt(model="rbf").fit(tsd)
      change_points = detector.predict(pen=30) #penalty
      
      changes_df = price_data[price_data.index.isin(change_points)]
      changes_df.style.set_caption("Sample of the Bitcoin Price change points in time series"). \
      set_properties(**{'border': '1.3px solid blue',
                                'color': 'grey'})
<img width="596" alt="image" src="https://github.com/user-attachments/assets/24feb39d-5acb-4966-a038-c79ce168d0a7">

- There are full of anomalies, indicating the market of Bitcoin is quite volatile and unstable.

      anomalies = results[results['Anomaly'] == 1]
      anomalies.head(30).style.set_caption("Sample of the Bitcoin Price anomalies in Time Series").set_properties(**{'border': '1.3px solid blue',
                                'color': 'grey'})

      results = results.set_index('Date')
        fig = px.line(results, x=results.index, y="Close",
                    title='Bitcoin Prices - Unsupervised Anomaly Detection',
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    template = 'plotly_white')
  
      outlier_dates = results[results['Anomaly'] == 1].index

      y_values = [results.loc[i]['Close'] for i in outlier_dates]
      fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers',
                      name = 'Anomaly',
                      marker=dict(color='red',size=5)))
      
      fig.show()

<img width="578" alt="image" src="https://github.com/user-attachments/assets/8d8a9d77-4fb5-4116-b483-7d44e822c102">
   
## 6. Hypotheses
Predicting Bitcoin prices is a complex task, and several machine learning models are commonly used for this purpose. Here are the top 2 most popular models for Bitcoin price prediction, along with their advantages and disadvantages:

| **Model**         | **Advantages**                                         | **Disadvantages**                                    |
|-------------------|-------------------------------------------------------|------------------------------------------------------|
| **ARIMA**         | Simple, interpretable, good for short-term forecasting, minimal data requirements | Struggles with non-stationary data, only models linear relationships, not suitable for long-term |
| **LSTM**          | Captures long-term dependencies, handles non-stationary data, adapts to complex data | Requires large data, complex, risk of overfitting     |

Each of these models has its strengths and limitations depending on the context of prediction (short-term vs. long-term, available data, etc.). For Bitcoin price prediction, LSTM tends to perform better in volatile, time-dependent situations, while ARIMA can be effective for short-term linear trends. In this project, we'll try to predict Bitcoin's price using both ARIMA and LTSM models.

When predicting Bitcoin prices using machine learning models like LSTM, it's important to choose the right price to model. In general, the closing price is the most commonly used price for prediction in financial forecasting, including Bitcoin, for several reasons:

- The closing price represents the final price at which Bitcoin was traded at the end of a given time period (e.g., daily, hourly).
- It is considered the most important price since it reflects the market consensus at the end of the trading session and is less volatile compared to other prices.
- Most financial analyses and forecasting models use the closing price as it tends to smooth out the noise and extreme fluctuations that can occur during the trading period.

### a. ARIMA model
Let’s take a look at ARIMA, which is one of the most popular (if not the most popular) time series forecasting techniques. In order to really understand ARIMA, we need to deconstruct its building blocks. Once we have the components down, it will become easier to understand how this time series forecasting method works as a whole.

Autoregressive (AR) part The Autoregressive (AR) component builds a trend from past values in the AR framework for predictive models. For clarification, the 'autoregression framework' works like a regression model where you use the lags of the time series' own past values as the regressors.

Integrated (I) part The Integrated (I) part involves the differencing of the time series component keeping in mind that our time series should be stationary, which really means that the mean and variance should remain constant over a period of time. Basically, we subtract one observation from another so that trends and seasonality are eliminated. By performing differencing we get stationarity. This step is necessary because it helps the model fit the data and not the noise.

Moving average (MA) part The moving average (MA) component focuses on the relationship between an observation and a residual error. Looking at how the present observation is related to those of the past errors, we can then infer some helpful information about any possible trend in our data.

We can consider the residuals among one of these errors, and the moving average model concept estimates or considers their impact on our latest observation. This is particularly useful for tracking and trapping short-term changes in the data or random shocks. In the (MA) part of a time series, we can gain valuable information about its behavior which in turn allows us to forecast and predict with greater accuracy.

Now let's try to build ARIMA model using our data.
- Split Data into Training and Out-of-Time Test Sets

      price_data = price_data.set_index('Date')
      price_data = price_data.sort_index()  # Ensure data is sorted by date
      price_data = price_data.asfreq('D')   # Set frequency if needed
      
      # Split data: 90% for training, 10% for out-of-time test
      close_price = price_data['Close']
      train_data_arima = close_price[:int(len(close_price) * 0.9)]
      out_of_time_test_data_arima = close_price[int(len(close_price) * 0.9):]
      
      #print(train_data_arima.shape, out_of_time_test_data_arima.shape)

- Differencing the Data to Make it Stationary

      train_data_arima_diff = train_data_arima.diff().dropna()

      # Use the Augmented Dickey-Fuller (ADF) test to check stationarity
      result = adfuller(train_data_arima_diff)
      print(f'ADF Statistic: {result[0]}')
      print(f'p-value: {result[1]}')
- Determining ARIMA Parameters (p, d, q)

      # Plot ACF and PACF
      fig, axes = plt.subplots(1, 2, figsize=(15, 5))
      plot_acf(train_data_arima_diff, lags=20, ax=axes[0])
      axes[0].set_title('ACF Plot')
      plot_pacf(train_data_arima_diff, lags=20, ax=axes[1])
      axes[1].set_title('PACF Plot')
      plt.show()
- Build and Train the ARIMA Model

      model = ARIMA(train_data_arima, order=(1, 1, 1))
      model_fit = model.fit()
      print(model_fit.summary())
- Make Predictions

      forecast_steps = len(out_of_time_test_data_arima)
      forecast = model_fit.forecast(steps=forecast_steps)
      
      # Align forecast index with test data for easier comparison
      forecast.index = out_of_time_test_data_arima.index
      
      # Plotting the actual vs predicted values
      plt.figure(figsize=(12, 6))
      plt.plot(price_data.index, price_data['Close'], label='Actual Bitcoin Price')
      plt.plot(forecast.index, forecast, label='Predicted Price', color='orange')
      plt.xlabel('Date')
      plt.ylabel('Price')
      plt.title('Bitcoin Price Prediction using ARIMA')
      plt.legend()
      plt.show()
<img width="548" alt="image" src="https://github.com/user-attachments/assets/7cd2992d-0917-4b34-a1fb-44d35fa9e67e">

- Evaluate Model Performance

      # Use actual values for the forecast period
      y_test = close_price[-forecast_steps:]  # Last known values to compare with
      
      # Use the forecasted values from ARIMA for comparison
      predictions = forecast
      
      # Calculate error metrics
      mse = mean_squared_error(y_test, predictions)
      mae = mean_absolute_error(y_test, predictions)
      rmse = np.sqrt(mse)
      mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
      
      print(f'MSE: {mse}')
      print(f'MAE: {mae}')
      print(f'RMSE: {rmse}')
      print(f'MAPE: {mape}')

MSE: 677382023.3422761
MAE: 21843.152912486807
RMSE: 26026.56380205186
MAPE: 37.680788951776876

Findings:
- A high MSE of about 677,382,023.34 suggests a significant difference between predicted and actual prices, which could reflect the model's struggle to capture volatile changes in Bitcoin prices.
- The MAE of approximately 21,843.15 implies a substantial average error, suggesting that on average, predictions deviate by 21,843 USD from the true values.
- The high RMSE of about 26,026.56 gives more weight to large errors, highlighting the presence of some substantial errors between predicted and actual values. This suggests the ARIMA model may be struggling with higher volatility in Bitcoin prices.
- A MAPE of 37.68% means the predictions are, on average, off by about 37.68% from the actual Bitcoin prices. Lower MAPE values (typically below 10%) are desirable in time series forecasting, and this relatively high MAPE indicates room for improvement in the model's forecasting accuracy.

Potential Reasons for High Error Metrics:
- Volatility of Bitcoin prices: Bitcoin prices are highly volatile, and ARIMA models may struggle to capture sharp, non-linear trends typical of financial data.
- Non-stationary trends: Bitcoin's price might have trends or cycles that simple differencing cannot account for, potentially making ARIMA insufficient without further adjustments.

Recommendation:
Adjustments, such as refining model parameters or considering additional features, could not help improve the forecasting accuracy. Consider alternative approaches, like LSTM, which might handle non-linearity and volatility better. Now let's try with LSTM model.

### b. LSTM model
Long Short Term Memory (LSTMs) is capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state.
Here's the steps to build LSTM model:

- Split the Data

      train_data = close_price[:int(len(close_price) * 0.7)]
      val_data = close_price[int(len(close_price) * 0.7):int(len(close_price) * 0.9)]
      out_of_time_test_data = close_price[int(len(close_price) * 0.9):]

- Normalize the Data

      scaler = MinMaxScaler()
      train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
      val_data_scaled = scaler.transform(val_data.values.reshape(-1, 1))
      out_of_time_test_scaled = scaler.transform(out_of_time_test_data.values.reshape(-1, 1))

- Create Sequences

      # Function to create sequences
      def create_sequences(data, sequence_length):
          X = []
          y = []
      
          for i in range(len(data) - sequence_length - 1):
              X.append(data[i:(i + sequence_length), 0])  # Extract values for consistency in numpy array
              y.append(data[i + sequence_length, 0])      # The next value following the sequence
      
          return np.array(X), np.array(y)
      
      # Sequence length
      sequence_length = 50
      
      # Generate sequences for each dataset
      X_train, y_train = create_sequences(train_data_scaled, sequence_length)
      X_val, y_val = create_sequences(val_data_scaled, sequence_length)
      X_test, y_test = create_sequences(out_of_time_test_scaled, sequence_length)
      
      # Reshape the sequences for LSTM input (samples, time steps, features)
      X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
      X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
      X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

- Build the LSTM Model

      #Define the LSTM model
      model_input = layers.Input(shape = X_train.shape[1:])
      model_lstm = layers.LSTM(15)(model_input)
      model_dense = layers.Dense(1)(model_lstm)
      model = models.Model(inputs=model_input, outputs=model_dense)
      
      # Compile the model
      model.compile(optimizer='adam', loss='mean_squared_error')
- Train the Model

      # Define early stopping callback
      early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
      
      # Train the model with early stopping
      history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])
      
      # Plot training history
      plt.plot(history.history['loss'], label='Train Loss')
      plt.plot(history.history['val_loss'], label='Validation Loss')
      plt.title('Model Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend()
      plt.show()

- Make Predictions

      #Make predictions on the out-of-time test set
      predictions = model.predict(X_test).reshape(-1, 1)  # Reshape to (n_samples, 1)
      
      # Inverse transform predictions and y_test
      predicted_price = scaler.inverse_transform(predictions)
      actual_price = scaler.inverse_transform(y_test.reshape(-1, 1))
      
      # 7. Visualize the actual vs predicted prices
      plt.plot(actual_price, label='Actual Price')  # Actual prices from test period
      plt.plot(predicted_price, label='Predicted Price')  # Predicted prices
      plt.title('Bitcoin Price Prediction')
      plt.xlabel('Time')
      plt.ylabel('Price')
      plt.legend()
      plt.show()

![image](https://github.com/user-attachments/assets/5ea8a61f-f11b-49da-abdf-7541973bd34d)

- Evaluate the Model

      #Calculate MSE, MAE, RMSE
      mse = mean_squared_error(actual_price, predicted_price)
      mae = mean_absolute_error(actual_price, predicted_price)
      rmse = math.sqrt(mse)
      
      print(f'MSE: {mse}')
      print(f'MAE: {mae}')
      print(f'RMSE: {rmse}')
      
      #Calculate MAPE
      def mean_absolute_percentage_error(y_true, y_pred):
          y_true, y_pred = np.array(y_true), np.array(y_pred)
          return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
      
      mape = mean_absolute_percentage_error(actual_price, predicted_price)
      print(f'MAPE: {mape}')

MSE: 5373281.25865455
MAE: 1743.9671140061214
RMSE: 2318.033920945625
MAPE: 3.043678277521293

Findings:

- This relatively lower MSE (5,373,281.26) compared to the ARIMA model suggests the LSTM model might be better capturing some of the patterns in the data.
- MAE (1,743.97) suggests that, on average, the LSTM model's predictions are around 1,744 USD away from the actual Bitcoin prices.
- RMSE (2,318.03) suggests the model’s error magnitude, with larger deviations from the actual prices more heavily penalized.
- A MAPE of 3.04% indicates that, on average, the model's predictions are off by just 3.04% of the actual Bitcoin prices. A lower MAPE, generally under 10%, is often seen as an indicator of high forecasting accuracy in time series models.

These metrics suggest that the LSTM model performs quite well, especially with a low MAPE (3.04%) indicating accurate predictions relative to actual prices. Lower values across these metrics, compared to other models, imply that the LSTM might be effectively capturing the patterns in the data, especially in a volatile and non-linear series like Bitcoin prices.

## 7. Recommendations
The study of Bitcoin’s historical price trends and the factors influencing its value reveals a complex interplay of market sentiment, regulatory developments, technological advancements, and macroeconomic conditions. Historically, Bitcoin’s price has experienced high volatility, marked by sharp rises and corrections, often influenced by external events, such as regulatory news or financial instability. Predicting Bitcoin’s future price with accuracy remains challenging due to its sensitivity to a range of unpredictable factors.

We can see how the above models perform with historic Bitcoin Price data. The prediction is not good. We need additional data from news or social media to make these models perform better and more accurately.

There are numerous time series models which can be used for forecasting time series data but the choice of the model totally depends on the problem statement. For example, the ARIMA modelling technique which we looked at in this blog is the simple time series technique that makes the predictions without taking into consideration other factors which might be affecting our dependent variable.

To enhance predictive accuracy, we recommend using a multi-model approach, incorporating both technical analysis (e.g., moving averages, trend lines) and machine learning models trained on historical and sentiment data. Additionally, given the market’s responsiveness to economic and regulatory news, monitoring real-time data and major policy shifts is crucial. Regular updates and continuous model validation will help improve prediction reliability as the cryptocurrency market evolves. Lastly, incorporating a risk management strategy is essential due to Bitcoin's inherent volatility, which can mitigate losses during periods of market downturn.




