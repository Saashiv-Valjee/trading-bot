import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from sklearn.model_selection import train_test_split

# Replace YOUR_ACCESS_TOKEN and YOUR_ACCOUNT_ID with your actual access token and account ID
access_token = "90c07764793c7d1818f3efa3fa997e3a-5ea6edea9acfe0b6821993523b9bce7c"
account_id = "101-004-24225231-003"

# Set up the Oanda API client
client = oandapyV20.API(access_token=access_token)

# Set the currency pair that you want to trade
currency_pair = "EUR_USD"

# Set the parameters for the trading bot
# You can adjust these parameters to suit your needs
buy_threshold = 1.05 # Buy if the price falls below this threshold
sell_threshold = 1.1 # Sell if the price rises above this threshold
amount_per_trade = 100 # The amount you want to spend on each trade
count = 1000 # how much historical data do we use 
interval = 10 # how long to wait at end of loop
window_size = 50 # size of windows for NN

# Function to place a buy order
def buy(price):
    price = float(price)
    margin_rate = 0.033
    units_to_buy = round(amount_per_trade / float(price))
    stop_loss_price = round(price - (price * (margin_rate / 100)),5)
    take_profit_price = round(price + (price * (margin_rate / 100)),5)
    data = {
        "order": {
            "price": str(price),
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "price": str(stop_loss_price),
            },
            "takeProfitOnFill": {
                "timeInForce": "GTC",
                "price": str(take_profit_price),
            },
            "instrument": "EUR_USD",
            "units": str(units_to_buy),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    print('BUYING')
    print(f"Price: {price},\nStop loss price: {stop_loss_price}\nTake profit price: {take_profit_price}")

    r = orders.OrderCreate(accountID=account_id, data=data)
    client.request(r)

# Function to place a sell order
def sell(price):
    # Retrieve the current account details
    r = accounts.AccountDetails(accountID=account_id)
    account_details = client.request(r)
    
    # Check if there are any units available to sell
    units_available = int(account_details['account']['units'])
    if units_available == 0:
        print("There are no units available to sell.")
        return
    
    price = float(price)
    # calculate the units to sell based on the amount of money you want to trade
    units_to_sell = round(amount_per_trade / float(price))

    # calculate the margin rate based on the amount of money you want to trade
    margin_rate = 0.33333333333333
    stop_loss_price = round(price + (price * (margin_rate / 100)),5)
    take_profit_price = round(price - (price * (margin_rate / 100)),5)

    # create the data for the sell order
    data = {
        "order": {
            "price": str(price),
            "stopLossOnFill": {
                "timeInForce": "GTC",
                "price": str(stop_loss_price)
            },
            "takeProfitOnFill": {
                "timeInForce": "GTC",
                "price": str(take_profit_price)
            },
            "instrument": "EUR_USD",
            "units": str(units_to_sell),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    # create the sell order
    r = orders.OrderCreate(accountID=account_id, data=data)

    print('SELLING')
    print(f"Price: {price},\nStop loss price: {stop_loss_price}\nTake profit price: {take_profit_price}")

    # send the request to the OANDA API
    client.request(r)

# Function to retrieve and preprocess historical data
def get_historical_data(count):
    # Use the Oanda API's InstrumentsCandles endpoint to retrieve 500 candles of historical data for the EUR/USD currency pair
    r = instruments.InstrumentsCandles(instrument=currency_pair, params={"count": count, 'granularity':'H4'})
    data = client.request(r)

    # Extract the close price from each candle and store it in a list
    close_prices = [candle["mid"]["c"] for candle in data["candles"]]

    # Return the list of close prices
    return close_prices

# Function to train and evaluate a machine learning model
def train_model(data, window_size):
    # Convert the close prices from strings to floats using numpy
    close_prices = np.array(data, dtype=float)
    print(data)

    # Split the close prices into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(close_prices, close_prices, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    # Define a simple neural network with a single output neuron
    model = models.Sequential()
    model.add(layers.Dense(units=1, input_dim=1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model on the training data and evaluate it on the validation data
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val, y_val))

    return model


# Main loop
while True:
    # Retrieve and preprocess the historical data
    print('retrieving data')
    data = get_historical_data(count)

    # Train and evaluate the machine learning model
    print('training model')
    data = np.array(data)
    model = train_model(data, window_size)

    # Use the model to make a prediction about the next price movement
    print('predicting price')
    prediction = model.predict(data.astype(float))
    print(data[-10:],prediction[:100])
    # Get the current market price for the currency pair
    print('gathering current market price for currecny pair')
    r = instruments.InstrumentsCandles(instrument=currency_pair, params={"count": 1, "granularity": "M1"})
    dataC = client.request(r)
    current_price = float(dataC["candles"][0]["mid"]["c"])

    # Adjust the buy_threshold and sell_threshold based on the prediction
    # You can adjust the coefficients (0.5 and 1.5 in this example) to suit your needs
    prediction_mean = np.mean(prediction)
    print(f'current price: {current_price}, prediction price: {prediction_mean}')
    if prediction_mean > float(current_price):
        print('predicting price increase')
        buy_threshold = current_price * 0.5
        sell_threshold = current_price * 1.5
    else:
        print('predicting price decrease or static price')
        buy_threshold = current_price * 1.5
        sell_threshold = current_price * 0.5

    # Place a buy order if the current price is below the buy threshold
    print(f'current price:{current_price}, buy order:{buy_threshold}, sell order:{sell_threshold}')
    if current_price < buy_threshold:
        buy(current_price)

    # Place a sell order if the current price is above the sell threshold
    elif current_price > sell_threshold:
        sell(current_price)

    # Wait for one minute before making the next trade
    print(f'------waiting {interval} seconds------')
    time.sleep(interval)
