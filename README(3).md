# **Tesla Stock Price Prediction Using LSTM**



### Project Overview



This project focuses on time series forecasting of Tesla (TSLA) stock prices using a Long Short-Term Memory (LSTM) neural network. LSTM models are well-suited for sequential data and are capable of learning long-term dependencies, making them effective for financial time series prediction.



The goal of this project is to:



* Analyze historical Tesla stock price data
* Preprocess and structure the data for deep learning
* Build and train an LSTM-based forecasting model
* Evaluate model performance using standard regression metrics
* Visualize predicted prices against actual prices



### Dataset Information



Dataset Source: Publicly available historical Tesla (TSLA) stock price dataset



Stock Symbol: TSLA (Tesla Inc.)



Features Used:



* Open
* High
* Low
* Close
* Volume
* Target Variable: Closing Price (Close)



The dataset contains historical daily stock prices, which are ideal for sequential modeling.



### Data Preprocessing Steps



The following preprocessing steps were applied before training the model:



* Handling Missing Values
* Checked for null or missing values.
* Removed or handled inconsistencies where necessary.
* Feature Selection
* Selected relevant numerical features.
* Focused primarily on the Close price for prediction.
* Data Scaling
* Applied MinMaxScaler to normalize values between 0 and 1.
* Scaling is essential for stable and faster LSTM training.
* Sequence Creation
* Converted the dataset into time-step sequences.
* Each input sequence consists of previous n days used to predict the next day’s price.
* Train-Test Split
* Training data: 75%
* Testing data: 25%
* Maintained chronological order (no shuffling).



### LSTM Model Architecture



The LSTM model was designed with the following architecture:



* Input Layer
* Accepts time-series sequences as input.
* LSTM Layers
* One or more stacked LSTM layers.
* Capable of capturing temporal dependencies in stock price movements.
* Dropout Layers
* Added to prevent overfitting.
* Dense Output Layer
* Single neuron output for predicting the next closing price.



### Model Summary



Optimizer: Adam



Loss Function: Mean Squared Error (MSE)



Epochs: Configurable (e.g., 15–50)



Batch Size: Typically 32 or 64



### Steps to Run the Project



1.Clone the Repository



git clone https://github.com/Abinaya-2117/Tesla\_stock\_prediction\_model.git

cd Tesla\_stock\_prediction\_model



2.Install Dependencies

pip install -r requirements.txt



Required Libraries:



* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* tensorflow / keras



3.Run the Notebook



jupyter notebook TeslaStockPricePredictionModel.ipynb



Execute all cells sequentially to:



1. Preprocess data
2. Train the LSTM model
3. Evaluate performance
4. Visualize results



### Model Evaluation Metrics



The model’s performance was evaluated using the following metrics:



* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)



These metrics help measure the difference between actual stock prices and predicted values.



### Results \& Visualizations



* The model successfully learned overall trends in Tesla’s stock price.
* Predictions closely follow actual price movements during stable periods.
* Slight deviations occur during high volatility, which is common in financial forecasting.



### Visualization Includes:



1. Actual vs Predicted Closing Prices
2. Trend comparison plots



These plots clearly demonstrate how well the LSTM model captures temporal price patterns.



### Key Observations \& Challenges



#### Observations



* LSTM performs well for short-term trend prediction.
* Data scaling significantly improves convergence.
* Increasing time steps improves contextual learning.



#### Challenges



* Stock prices are influenced by external factors (news, events).
* High volatility reduces prediction accuracy.
* Overfitting risk with deeper networks.



#### Potential Improvements



* Incorporate technical indicators (RSI, MACD, Moving Averages)
* Use multivariate LSTM models
* Apply hyperparameter tuning
* Experiment with GRU or Transformer-based models
* Include sentiment analysis from news or social media



### Supporting Materials 



Graphs and charts of predictions vs actual prices



### Conclusion



This project demonstrates how LSTM neural networks can be effectively applied to stock price forecasting. While no model can perfectly predict financial markets, the results show strong potential for learning historical trends and making informed predictions.

