from build_proxy import build_proxy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras import models, layers, optimizers
from scikeras.wrappers import KerasRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor

def ffnn_model(hidden_layers=1, units=64, activation='relu', optimizer='adam', learning_rate=0.001, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Dense(units, input_dim=8, activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='linear'))  # Assuming a regression problem
    
    # Compile model
    if optimizer == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    # Add other optimizers if needed
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def get_results(dataset, selected_features, model, shuffle):
    x_dataset = dataset[selected_features]
    y_dataset = dataset["BC"]

    if shuffle:
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=42, shuffle=True)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, shuffle=False)
    # x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, shuffle=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse

def execution_SVR_models(hyperarameters, dataset, iteration, filename):
    c = hyperarameters["C"]
    gamma = hyperarameters["gamma"]
    epsilon = hyperarameters["epsilon"]
    shuffle = hyperarameters["shuffle"]

    svr_model_list = {
        "linear": SVR(kernel="linear", C=c, gamma=gamma, epsilon=epsilon),
        
        "poly": SVR(kernel="poly", C=c, gamma=gamma, epsilon=epsilon),
        
        "rbf": SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon),
    }

    for model_name, model in svr_model_list.items():
        selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
        r2, rmse = get_results(dataset, selected_features, model, shuffle)
        with open(filename, "a") as f:
            f.write(f"{model_name}; {shuffle}; {c}; {gamma}; {epsilon}; {r2}; {rmse}\n")
        print(f"{model_name}: {iteration}/72 done!")

def execution_RF_model(hyperarameters, dataset, iteration, filename):
    max_depth = hyperarameters["max_depth"]
    n_estimators = hyperarameters["n_estimators"]
    min_samples_split = hyperarameters["min_samples_split"]
    min_samples_leaf = hyperarameters["min_samples_leaf"]
    bootstrap = hyperarameters["bootstrap"]
    shuffle = hyperarameters["shuffle"]

    random_forest_model = RandomForestRegressor(
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        bootstrap=bootstrap
    )

    selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
    r2, rmse = get_results(dataset, selected_features, random_forest_model, shuffle)
    with open(filename, "a") as f:
        f.write(f"RF; {shuffle}; {max_depth}; {n_estimators}; {min_samples_split}; {min_samples_leaf}; {bootstrap}; {r2}; {rmse}\n")
    print(f"Random Forest {iteration}/324 done!")

def execution_FFNN_model(hyperarameters, dataset, iteration, filename):
    hyperarameters_ffnn = {
        'hidden_layers': [1, 2, 3],
        'units': [32, 64, 128],
        'learning_rate': [0.1, 0.01, 0.001],
        'epochs': [50, 100],
        'dropout_rate': [0.0, 0.2, 0.5],
        "shuffle": [True, False]
    }

    hidden_layers = hyperarameters["hidden_layers"]
    units = hyperarameters["units"]
    learning_rate = hyperarameters["learning_rate"]
    epochs = hyperarameters["epochs"]
    dropout_rate = hyperarameters["dropout_rate"]
    shuffle = hyperarameters["shuffle"]

    model = KerasRegressor(
        build_fn=ffnn_model(
            hidden_layers=hidden_layers, 
            units=units, 
            activation="relu", 
            optimizer="adam", 
            learning_rate=learning_rate, 
            dropout_rate=dropout_rate
        ), 
        verbose=0
    )

    print(f"Model: RF in progress...")
    selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
    r2, rmse = get_results(dataset, selected_features, model, shuffle)
    with open(filename, "a") as f:
        # f.write(f"RF; {max_depth}; {n_estimators}; {min_samples_split}; {min_samples_leaf}; {max_features}; {bootstrap}; {r2}; {rmse}\n")
        f.write(f"RF; {shuffle}; {hidden_layers}; {units}; {learning_rate}; {epochs}; {dropout_rate}; {r2}; {rmse}\n")
    print(f"FFNN {iteration}/324 done!")

def test_cases_SVR(dataset):
    hyperarameters_SVR = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "epsilon": [0.1, 0.3, 0.5],
        "shuffle": [True, False]
    }
    filename = f"final_results_SVR-r2-rmse.csv"
    with open(filename, "w") as f:
        # f.write("model; r2; rmse; degree; \n")
        f.write("model; shuffled; c; gamma; epsilon; r2; rmse\n")

    i = 1
    for c in tqdm(hyperarameters_SVR["C"], desc="Executing SVR models..."):
        for g in hyperarameters_SVR["gamma"]:
            for e in hyperarameters_SVR["epsilon"]:
                for s in hyperarameters_SVR["shuffle"]:
                    hyperarameters = {
                        "C": c,
                        "gamma": g,
                        "epsilon": e,
                        "shuffle": s
                    }
                    execution_SVR_models(hyperarameters, dataset, i, filename)
                    i += 1

def test_cases_RF(dataset):
    hyperarameters_RF = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [10, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'shuffle': [True, False]
    }

    filename = f"final_results_RF-r2-rmse.csv"
    with open(filename, "w") as f:
        f.write("model; shuffled; max_depth; n_estimators; min_samples_split; min_samples_leaf; bootstrap; r2; rmse\n")

    i = 1
    for n in tqdm(hyperarameters_RF["n_estimators"], desc="Executing RF models..."):
        for d in hyperarameters_RF["max_depth"]:
            for s in hyperarameters_RF["min_samples_split"]:
                for l in hyperarameters_RF["min_samples_leaf"]:
                    for b in hyperarameters_RF["bootstrap"]:
                        for sh in hyperarameters_RF["shuffle"]:
                            hyperarameters = {
                                'n_estimators': n,
                                'max_depth': d,
                                'min_samples_split': s,
                                'min_samples_leaf': l,
                                'bootstrap': b,
                                'shuffle': sh
                            }
                            execution_RF_model(hyperarameters, dataset, i, filename)
                            i += 1

def test_cases_FFNN(dataset):
    hyperarameters_ffnn = {
        'hidden_layers': [1, 2, 3],
        'units': [32, 64, 128],
        'learning_rate': [0.1, 0.01, 0.001],
        'epochs': [50, 100],
        'dropout_rate': [0.0, 0.2, 0.5],
        "shuffle": [True, False]
    }
    warnings.filterwarnings("ignore")

    filename = f"final_results_FFNN-r2-rmse.csv"
    with open(filename, "w") as f:
        # f.write("model; max_depth; n_estimators; min_samples_split; min_samples_leaf; max_features; bootstrap; r2; rmse\n")
        f.write("model; shuffled; hidden_layers; units; learning_rate; epochs; dropout_rate; r2; rmse\n")

    i = 1
    for h in tqdm(hyperarameters_ffnn["hidden_layers"], desc="Executing FFNN models..."):
        for u in hyperarameters_ffnn["units"]:
            for l in hyperarameters_ffnn["learning_rate"]:
                for e in hyperarameters_ffnn["epochs"]:
                    for d in hyperarameters_ffnn["dropout_rate"]:
                        for s in hyperarameters_ffnn["shuffle"]:
                            hyperarameters = {
                                'hidden_layers': h,
                                'units': u,
                                'learning_rate': l,
                                'epochs': e,
                                'dropout_rate': d,
                                'shuffle': s
                            }
                            execution_FFNN_model(hyperarameters, dataset, i, filename)
                            i += 1
    
def execution_GP_model(hyperarameters, dataset, iteration, filename):
    n_restarts_optimizer = hyperarameters["n_restarts_optimizer"]
    alpha = hyperarameters["alpha"]
    normalize_y = hyperarameters["normalize_y"]
    shuffle = hyperarameters["shuffle"]
    kernel = hyperarameters["kernel"]

    gaussian_process_model = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=n_restarts_optimizer, 
        alpha=alpha, 
        normalize_y=normalize_y
    )

    selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
    r2, rmse = get_results(dataset, selected_features, gaussian_process_model, shuffle)
    with open(filename, "a") as f:
        f.write(f"RF; {shuffle}; {kernel}; {n_restarts_optimizer}; {alpha}; {normalize_y}; {r2}; {rmse}\n")
    print(f"Gaussian {iteration}/108 done!")

def test_cases_GP(dataset):
    kernels = {
        "RBF": RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e5)),
        "Matern": Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e5), nu=1.5),
        "RationalQuadratic": RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds=(1e-6, 1e5))
    }

    hyperarameters_GP = {
        'n_restarts_optimizer': [0, 5, 10],
        'alpha': [1e-10, 1e-5, 1e-2],
        'normalize_y': [True, False],
        'shuffle': [True, False]
    }
    warnings.filterwarnings("ignore")

    filename = f"final_results_GP-r2-rmse.csv"
    with open(filename, "w") as f:
        # f.write("model; shuffled; n_restart_optimizer; alpha; normalize_y; r2; rmse\n")
        f.write("model; shuffled; kernel; n_restart_optimizer; alpha; normalize_y; r2; rmse\n")

    i = 1
    for kernel_name, kernel in tqdm(kernels.items(), desc="Executing GP models..."):
        for n in hyperarameters_GP["n_restarts_optimizer"]:
            for a in hyperarameters_GP["alpha"]:
                for ny in hyperarameters_GP["normalize_y"]:
                    for s in hyperarameters_GP["shuffle"]:
                        hyperarameters = {
                            'kernel': kernel,
                            'n_restarts_optimizer': n,
                            'alpha': a,
                            'normalize_y': ny,
                            'shuffle': s
                        }
                        execution_GP_model(hyperarameters, dataset, i, filename)
                        i += 1

def execution_KR_model(hyperarameters, dataset, iteration, filename):
    kernel = hyperarameters["kernel"]
    alpha = hyperarameters["alpha"]
    gamma = hyperarameters["gamma"]
    shuffle = hyperarameters["shuffle"]

    kernel_ridge = KernelRidge(
        kernel=kernel, 
        alpha=alpha, 
        gamma=gamma
    )

    selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
    r2, rmse = get_results(dataset, selected_features, kernel_ridge, shuffle)
    with open(filename, "a") as f:
        f.write(f"RF; {shuffle}; {kernel}; {alpha}; {gamma}; {r2}; {rmse}\n")
    print(f"Kernel Ridge {iteration}/120 done!")

def test_cases_KR(dataset):
    hyperarameters_KR = {
        'alpha': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': [None, 0.01, 0.1, 1, 10],  # Note: None is included for linear kernel, where gamma is not used
        'shuffle': [True, False]
    }

    filename = f"final_results_KR-r2-rmse.csv"
    with open(filename, "w") as f:
        # f.write("model; shuffled; n_restart_optimizer; alpha; normalize_y; r2; rmse\n")
        # f.write("model; shuffled; kernel; n_restart_optimizer; alpha; normalize_y; r2; rmse\n")
        f.write("model; shuffled; kernel; alpha; gamma; r2; rmse\n")

    i = 1
    for kernel in tqdm(hyperarameters_KR["kernel"], desc="Executing KR models..."):
        for a in hyperarameters_KR["alpha"]:
            for g in hyperarameters_KR["gamma"]:
                for s in hyperarameters_KR["shuffle"]:
                    hyperarameters = {
                        'kernel': kernel,
                        'alpha': a,
                        'gamma': g,
                        'shuffle': s
                    }
                    execution_KR_model(hyperarameters, dataset, i, filename)
                    i += 1

def execution_KNN_model(hyperarameters, dataset, iteration, filename):
    n_neighbors = hyperarameters["n_neighbors"]
    weights = hyperarameters["weights"]
    algorithm = hyperarameters["algorithm"]
    leaf_size = hyperarameters["leaf_size"]
    shuffle = hyperarameters["shuffle"]

    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size
    )

    selected_features = ['N_CPC', 'PM_1_0', 'O3', 'TEMP', 'HUM', 'PM_10', 'CO', 'NO2']
    r2, rmse = get_results(dataset, selected_features, knn, shuffle)
    with open(filename, "a") as f:
        f.write(f"KNN; {shuffle}; {n_neighbors}; {weights}; {algorithm}; {leaf_size}; {r2}; {rmse}\n")

    print(f"KNN {iteration}/320 done!")

def test_cases_KNN(dataset):
    hyperarameters_KNN = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40, 50],
        'shuffle': [True, False]
    }

    filename = f"final_results_KNN-r2-rmse.csv"
    with open(filename, "w") as f:
        f.write("model; shuffled; n_neighbors; weights; algorithm; leaf_size; r2; rmse\n")

    i = 1
    for n in tqdm(hyperarameters_KNN["n_neighbors"], desc="Executing KNN models..."):
        for w in hyperarameters_KNN["weights"]:
            for a in hyperarameters_KNN["algorithm"]:
                for l in hyperarameters_KNN["leaf_size"]:
                    for s in hyperarameters_KNN["shuffle"]:
                        hyperarameters = {
                            'n_neighbors': n,
                            'weights': w,
                            'algorithm': a,
                            'leaf_size': l,
                            'shuffle': s
                        }
                        execution_KNN_model(hyperarameters, dataset, i, filename)
                        i += 1

if __name__ == '__main__':
    dataset = build_proxy()
    test_cases_SVR(dataset)
    test_cases_RF(dataset)
    test_cases_FFNN(dataset)
    test_cases_GP(dataset)
    test_cases_KR(dataset)
    test_cases_KNN(dataset)
        
