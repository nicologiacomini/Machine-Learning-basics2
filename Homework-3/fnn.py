import numpy as np
from keras import models, layers, optimizers, regularizers
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from build_proxy import build_proxy
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Function to create model, required for KerasRegressor
def create_model(hidden_layers=1, units=64, activation='relu', optimizer='adam', learning_rate=0.001, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Dense(units, input_dim=8, activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(models.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='linear'))  # Assuming a regression problem
    
    # Compile model
    if optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    # Add other optimizers if needed
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Define the grid search parameters
param_grid = {
    'hidden_layers': [1, 2, 3],
    'units': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100],
    'dropout_rate': [0.0, 0.2, 0.5]
}

# Wrap the Keras model for use in scikit-learn
model = KerasRegressor(
    build_fn=create_model(hidden_layers=param_grid['hidden_layers'][0], units=param_grid['units'][0], activation=param_grid['activation'][0], optimizer=param_grid['optimizer'][0], learning_rate=param_grid['learning_rate'][0], dropout_rate=param_grid['dropout_rate'][0]), 
    verbose=0
)


# Fit the model (this will take some time as it's doing a grid search)
# Assuming X_train and y_train are your training data
dataset = build_proxy()
selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
x_dataset = dataset[selected_features]
y_dataset = dataset["BC"]

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2: {r2}")
print(f"RMSE: {rmse}")
