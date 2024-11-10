# Time Series Anomaly Detection via LSTM Autoencoder: A Predictive Analytics Framework

# List of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Project Setup](#project-setup)
5. [How to Run The Code](#how-to-run-the-code)
6. [Code Overview](#code-overview)
7. [Future Enhancements](#future-enhancements)

    
## Project Overview  
This project performs anomaly detection on financial time-series data (e.g., stock, cryptocurrency, and forex prices) using Long Short-Term Memory (LSTM) Autoencoders. The system identifies anomalies by measuring the reconstruction error when recreating sequences of historical price data, flagging high-error predictions as potential anomalies. Users can fetch time-series data from Yahoo Finance, preprocess it, and analyze anomalies via interactive visualizations.  

## Features  
- Fetch financial time-series data for cryptocurrencies, forex, or stocks.
- Normalize and preprocess data into sequences for temporal analysis.
- Train an LSTM Autoencoder to reconstruct normal behavior in time series.
- Calculate a dynamic threshold for anomaly detection based on the training error distribution.
- Visualize data trends, reconstruction error, and anomalies.  


## Requirements  
To run this project, ensure you have the following dependencies installed:  
```
plaintext  
numpy
pandas
tensorflow
seaborn
matplotlib
plotly
yfinance
scikit-learn
google-colab
```

You can install all dependencies using:
``` 
pip install -r requirements.txt
```

## Project Setup

### 1\. Data Fetching and Storage

The project uses Yahoo Finance data for market prices. When run in Google Colab, it automatically mounts Google Drive, creating a folder to store downloaded data. Users can select the type of asset (cryptocurrency, forex, or stock) and specify the folder name within Google Drive.

### 2\. Data Processing and Visualization

*   The fetch\_and\_save\_data function saves market data in CSV format, focusing on the Date and Close columns.
    
*   Data is visualized using Plotly, displaying trends over time for market prices.
    

### 3\. Training the LSTM Autoencoder

*   After data loading and visualization, the data is split into training (80%) and testing (20%) sets.
    
*   A StandardScaler normalizes the training data for consistent scaling.
    
*   Data sequences of 30 time steps each are created for the LSTM Autoencoder model to capture patterns over time effectively.
    

### 4\. Model Training

The model architecture consists of:

*   LSTM layers with dropout to avoid overfitting.
    
*   A RepeatVector layer to reconstruct input sequences.
    
*   A TimeDistributed Dense layer for element-wise predictions.
    

The model trains for up to 100 epochs with early stopping to prevent overfitting.

### 5\. Anomaly Detection

*   The mean absolute error (MAE) is calculated for both the training and test data.
    
*   A dynamic threshold is computed using the 95th percentile of the training error.
    
*   Data points with test MAE above this threshold are flagged as anomalies.
    

### 6\. Visualization of Anomalies

*   Anomalies are visualized as markers overlaid on the time-series data, making them easy to identify and interpret.
    

## How to Run the Code


1.  **Google Drive Mounting**:
    
    *   The script requires Google Colab with access to Google Drive for storage. Ensure you provide the appropriate permissions when prompted.
        
2.  **Fetching Market Data**:
    
    *   Run get\_user\_input() to prompt the user to select the asset type and enter details such as ticker, folder name, and other parameters.
        
3.  **Training and Anomaly Detection**:
    
    *   Once the data is fetched, it is preprocessed and used to train the LSTM Autoencoder. Anomalies are detected in the test set based on the dynamic threshold.
        
4.  **Viewing Results**:
    
    *   The results, including time series trends and anomaly markers, are visualized using Plotly and Seaborn, with interactive plots displayed inline.
        

Code Overview
-------------

### Data Fetching
```   
def fetch_and_save_data(asset_type, asset_pair, start_date, end_date, folder_name, interval='1d'):      # Function to fetch and save data from Yahoo Finance        
```

### Data Preprocessing

Converts raw data into standardized sequences using:
```  
scaler = StandardScaler()  train['Close'] = scaler.transform(train[['Close']])  test['Close'] = scaler.transform(test[['Close']])
 ```

### Model Definition
```   
pythonCopy codemodel = Sequential()  model.add(LSTM(128, input_shape=(timesteps, num_features)))  model.add(Dropout(0.2))  model.add(RepeatVector(timesteps))  model.add(LSTM(128, return_sequences=True))  model.add(Dropout(0.2))  model.add(TimeDistributed(Dense(num_features)))
```

### Training and Evaluation

Trains with early stopping and calculates MAE for anomaly detection:

```   
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')  history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], shuffle=False)
```

### Anomaly Visualization

Plots anomalies based on calculated MAE:
```
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:]['Date'], y=test_score_df['loss'], mode='lines', name='Test Loss'))
fig.add_trace(go.Scatter(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close'].values.reshape(-1, 1)), mode='markers', name='Anomaly'))
```

Future Enhancements
-------------------

1.  **Enhanced Data Sources**: Include other data points (e.g., sentiment analysis or macroeconomic indicators) to enhance anomaly context.
    
2.  **User-Defined Thresholds**: Add options for customized thresholds for different use cases.
    
3.  **Model Optimization**: Experiment with attention layers or other model architectures for improved accuracy.
