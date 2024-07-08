import stat_analysis as sa
from sklearn.svm import SVR
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from scikeras.wrappers import KerasRegressor
from tqdm import tqdm
from keras import models, layers, optimizers, regularizers

def build_proxy():
    dataset = sa.get_datasets()
    normalized_dataset = sa.normalization(dataset)
    return normalized_dataset

def ffnn():
    # create a sequential model
    model = models.Sequential()

    # add the hidden layer
    model.add(layers.Dense(input_dim=20,
                        units=10, 
                        activation="relu"))

    # add the output layer
    model.add(layers.Dense(input_dim=10,
                        units=1,
                        activation='sigmoid'))

    # define our loss function and optimizer
    model.compile(loss='binary_crossentropy',
                # Adam is a kind of gradient descent
                optimizer=optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])
    return model

def forward_subset_selection(dataset, model):
    selected_features_list = []

    x_dataset = dataset.drop(columns=["BC", "date"])
    y_dataset = dataset["BC"]

    for n_feat in range(1, 12):
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_feat, direction='forward', scoring='r2', cv=5)
        try:
            sfs.fit(x_dataset, y_dataset)
            selected_features = sfs.get_support(indices=True)
            print(f"\n\n\n\n\n\n\n\nSelected features: {selected_features}\n\n\n\n\n\n\n\n")
            
            selected_features = x_dataset.columns[sfs.get_support()]
            selected_features_list.append(selected_features)
        except Exception as e:
            print(e)
    
    return selected_features_list

def forward_neural_network():
    model_to_use = KerasRegressor(build_fn=ffnn, epochs=50, batch_size=10, verbose=0)
    return model_to_use

def get_results(dataset, selected_features, model):
    x_dataset = dataset[selected_features]
    y_dataset = dataset["BC"]

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse

if __name__ == '__main__':
    dataset = build_proxy()
    model_list = {
        "linear": SVR(kernel="linear"),
        
        "poly": SVR(kernel="poly"),
        
        "rbf": SVR(kernel="rbf"),
        "sigmoid": SVR(kernel="sigmoid"),
        
        "random forest": RandomForestRegressor(),
        # "ffnn": forward_neural_network()
    }

    with open("performance_results.csv", "w") as f:
        f.write("model; features; r2; rmse\n")

    for model_name, model in tqdm(model_list.items(), desc="Model execution"):
        print(f"Model: {model_name} in progress...")
        selected_features_list = forward_subset_selection(dataset, model)
        for selected_features in selected_features_list:
            print("...", end="")
            r2, rmse = get_results(dataset, selected_features, model)
            with open("performance_results.csv", "a") as f:
                f.write(f"{model_name}; {selected_features}; {r2}; {rmse}\n")
        print(f"\nModel: {model_name} done!")