

# Bitcoin Analysis V2

This script performs a comprehensive analysis of Bitcoin price data using various data manipulation and visualization techniques. The analysis includes data preprocessing, exploratory data analysis (EDA), and visualization using libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Plotly.
# Results
The analysis of Bitcoin price data reveals significant fluctuations and trends over time. Visualizations of open, high, low, and close prices show the volatility of Bitcoin prices. The candlestick chart provides a detailed view of the price movements, highlighting periods of high and low activity. The closing price trends, when analyzed on yearly, quarterly, and monthly bases, show distinct patterns and cycles. Log scaling of the closing prices helps in better understanding the price changes over time. The daily price change analysis indicates the percentage change in closing prices, providing insights into the daily volatility of Bitcoin.
## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Loading the Data](#loading-the-data)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Visualization](#visualization)
7. [Detailed Analysis of Closing Prices](#detailed-analysis-of-closing-prices)
8. [Usage](#usage)
9. [Contributing](#contributing)


## Introduction

This script analyzes Bitcoin price data to understand the trends and patterns in the price movements over time. The analysis includes data preprocessing, exploratory data analysis, and visualization using various Python libraries.

## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- matplotlib
- seaborn
- sqlite3
- plotly
- chart_studio

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn sqlite3 plotly chart_studio
```

## Loading the Data

The dataset is loaded from a CSV file using Pandas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Load the dataset
df = pd.read_csv(r'N:\Personal_Projects\Bitcoin\bitcoin_price_Training - Training.csv')
```

## Data Preprocessing

The data preprocessing steps include converting the 'Date' column to datetime format, checking for null values, and removing duplicates.

### Displaying Initial Data

```python
# Display the first and last few rows of the dataset
print(df.head())
print(df.tail())

# Display the columns and shape of the dataset
print(df.columns)
print(df.shape)

# Display the data types and summary statistics of the dataset
print(df.info())
print(df.describe())
print(df.describe().T)
print(df.dtypes)
```

### Converting Date Column

The 'Date' column is initially in string format. It needs to be converted to a datetime format for time series analysis.

```python
# Convert the 'Date' column to datetime format
df['Date'] = df['Date'].astype('datetime64[ns]')

# Display the minimum and maximum dates
print(df['Date'].min())
print(df['Date'].max())
```

### Checking for Null Values and Duplicates

```python
# Check for null values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())
```

### Sorting and Resetting Index

The dataset is sorted by index in descending order and the index is reset.

```python
# Sort the dataset by index in descending order and reset the index
data = df.sort_index(ascending=False).reset_index()

# Drop the 'index' column
data.drop('index', axis=1, inplace=True)
```

## Exploratory Data Analysis (EDA)

The exploratory data analysis includes visualizing the price trends over time and analyzing the open, high, low, and close prices.

### Visualizing Price Trends

```python
# Plot the price trends over time
plt.figure(figsize=(20, 12))
for index, col in enumerate(['Open', 'High', 'Low', 'Close'], 1):
    plt.subplot(2, 2, index)
    plt.plot(df['Date'], df[col])
    plt.title(col)
```

### Creating a Sample Dataset

```python
# Display the shape of the dataset
print(data.shape)

# Create a sample dataset for visualization
bitcoin_sample = data[0:50]
```

## Visualization

The visualization includes creating a candlestick chart using Plotly to analyze the open, high, low, and close prices.

### Installing Plotly and Chart Studio

```python
# Install Plotly and Chart Studio
!pip install chart_studio
!pip install plotly
```

### Importing Plotly and Initializing Offline Mode

```python
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Initialize Plotly in offline mode
init_notebook_mode(connected=True)
```

### Creating a Candlestick Chart

```python
# Create a candlestick chart
trace = go.Candlestick(x=bitcoin_sample['Date'], high=bitcoin_sample['High'],
                       open=bitcoin_sample['Open'], close=bitcoin_sample['Close'],
                       low=bitcoin_sample['Low'])

candle_data = [trace]
fig = go.Figure(data=candle_data)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()
```

### Adding a Title to the Chart

```python
# Add a title to the chart
layout = {
    'title': 'Bitcoin Historical Price',
    'xaxis': {'title': 'Date'}
}
fig = go.Figure(data=candle_data, layout=layout)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()
```

## Detailed Analysis of Closing Prices

The detailed analysis of closing prices includes visualizing the closing price trends, applying log scaling, and analyzing the yearly, quarterly, and monthly trends.

### Visualizing Closing Price Trends

```python
# Plot the closing price trends
data['Close'].plot()

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Plot the closing price trends with log scaling
plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
data['Close'].plot()
plt.title('No scaling')
plt.subplot(1, 2, 2)
np.log1p(data['Close']).plot()
plt.title('Log scaling')
plt.yscale('log')
```

### Analyzing Yearly, Quarterly, and Monthly Trends

```python
# Analyze the closing price trends on a yearly, quarterly, and monthly basis
print(data['Close'].resample('Y').mean())
data['Close'].resample('Y').mean().plot()
print(data['Close'].resample('Q').mean())
data['Close'].resample('Q').mean().plot()
```

### Analyzing Daily Price Change

```python
# Analyze the daily price change in closing prices
data['Close_pct_change'] = data['Close'].pct_change() * 100
data['Close_pct_change'].plot()
```

## Usage

To use the code, clone the repository and run the Python script. Make sure to have the required dependencies installed.

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
python your_script.py
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

---

