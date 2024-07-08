import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def get_datasets():
    # date;BC;N_CPC;PM-10;PM-2.5;PM-1.0;NO2;O3;SO2;CO;NO;NOX;TEMP;HUM
    date = []
    BC = []
    N_CPC = []
    PM_10 = []
    PM_2_5 = []
    PM_1_0 = []
    NO2 = []
    O3 = []
    SO2 = []
    CO = []
    NO = []
    NOX = []
    TEMP = []
    HUM = []

    isFirst = True
    with open('BC-Data-Set.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if isFirst:
                isFirst = False
                continue
            date.append(row[0])
            BC.append(float(row[1]))
            N_CPC.append(float(row[2]))
            PM_10.append(float(row[3]))
            PM_2_5.append(float(row[4]))
            PM_1_0.append(float(row[5]))
            NO2.append(float(row[6]))
            O3.append(float(row[7]))
            SO2.append(float(row[8]))
            CO.append(float(row[9]))
            NO.append(float(row[10]))
            NOX.append(float(row[11]))
            TEMP.append(float(row[12]))
            HUM.append(float(row[13]))
    
    dataset = {
        "date": date,
        "BC": BC,
        "N_CPC": N_CPC,
        "PM_10": PM_10,
        "PM_2_5": PM_2_5,
        "PM_1_0": PM_1_0,
        "NO2": NO2,
        "O3": O3,
        "SO2": SO2,
        "CO": CO,
        "NO": NO,
        "NOX": NOX,
        "TEMP": TEMP,
        "HUM": HUM
    }

    dataset = pd.DataFrame(dataset)
    return dataset

def get_stats(dataset):
    with open("output/stats-norm.csv", "w") as f:
        f.write(f"Dataset; mean; median; std; var; min; max\n")
        for column in dataset.columns:
            if column == "date":
                continue
            data_column = dataset[column]
            mean = data_column.mean()
            median = data_column.median()
            std = data_column.std()
            var = data_column.var()
            min = data_column.min()
            max = data_column.max()
            f.write(f"{column}; {mean}; {median}; {std}; {var}; {min}; {max}\n")

def normalization(dataset):
    for column in dataset.columns:
        if column == "date":
            continue
        data_column = dataset[column]
        data_column = (data_column - data_column.mean()) / abs(data_column.std())
        # data_column = (data_column - data_column.min()) / (data_column.max() - data_column.min())
        dataset[column] = data_column
    return dataset

def show_plots(dataset):
    colors = [
        "Red",
        "Green",
        "Blue",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Brown",
        "Black",
        "White",
        "Gray",
        "Cyan",
        "Magenta"
    ]

    plt.figure(figsize=(10, 8))
    i = 0
    max = 100
    n_measurements = list(range(max))
    for column in dataset.columns:
        if column == "date":
            continue
        plt.plot(n_measurements, dataset[column][0:max], color=f'{colors[i]}', label=f'{column}', linewidth=1)
        # plt.plot(n_measurements, dataset[column], color=f'{colors[i]}', label=f'{column}', linewidth=1)
        i += 1
    plt.xlabel('Number of measurements')
    plt.ylabel('Values measured')
    plt.legend()
    plt.grid(True)  # Add grid
    plt.title('Visualization of all data normalized')
    plt.savefig("output/all_data_visualization.png")
    plt.close()

def correlation_matrix(dataset):
    new_dataset = dataset.drop(columns=['date'])
    
    # Pearson correlation
    correlation_matrix = new_dataset.corr('pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix Heatmap')
    plt.savefig("output/pearson-correlation.png")
    plt.close()
    correlation_matrix = None

    # Kendall correlation
    correlation_matrix = new_dataset.corr('kendall')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Kendall Correlation Matrix Heatmap')
    plt.savefig("output/kendall-correlation.png")
    plt.close()

    # Spearman correlation
    correlation_matrix = new_dataset.corr('spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation Matrix Heatmap')
    plt.savefig("output/spearman-correlation.png")
    plt.close()

def make_plot(dataset1, dataset2, name1, name2):
    plt.figure(figsize=(10, 8))
    plt.scatter(dataset1, dataset2, s=3, color='blue')
    plt.xlabel(name1)
    plt.ylabel(name2)
    M = max(dataset1.max(), dataset2.max())
    m = min(dataset1.min(), dataset2.min())
    plt.plot([m, M], [m, M], ls="--",c='red')
    plt.grid(True)  # Add grid
    plt.title(f'{name1} vs {name2}')
    plt.savefig(f"output/plot-{name1}-{name2}.png")
    plt.close()

def scatter_plots(dataset):
    # # Pm10 vs pm2.5
    # make_plot(dataset['PM_10'], dataset['PM_2_5'], 'PM_10', 'PM_2_5')
    # # Pm10 vs pm1.0
    # make_plot(dataset['PM_10'], dataset['PM_1_0'], 'PM_10', 'PM_1_0')
    # # Pm2.5 vs pm1.0
    # make_plot(dataset['PM_2_5'], dataset['PM_1_0'], 'PM_2_5', 'PM_1_0')
    # # No2 vs no
    # make_plot(dataset['NO2'], dataset['NO'], 'NO2', 'NO')
    # # No2 vs nox
    # make_plot(dataset['NO2'], dataset['NOX'], 'NO2', 'NOX')
    # # No vs nox
    # make_plot(dataset['NO'], dataset['NOX'], 'NO', 'NOX')
    # # No2 vs o3
    # make_plot(dataset['NO2'], dataset['O3'], 'NO2', 'O3')
    # # o3 vs nox
    # make_plot(dataset['O3'], dataset['NOX'], 'O3', 'NOX')
    # BC vs N_CPC
    make_plot(dataset['BC'], dataset['N_CPC'], 'BC', 'N_CPC')
    # BC vs PM_10
    make_plot(dataset['BC'], dataset['PM_10'], 'BC', 'PM_10')
    # BC vs PM_2_5
    make_plot(dataset['BC'], dataset['PM_2_5'], 'BC', 'PM_2_5')
    # BC vs PM_1_0
    make_plot(dataset['BC'], dataset['PM_1_0'], 'BC', 'PM_1_0')
    # BC vs NO2
    make_plot(dataset['BC'], dataset['NO2'], 'BC', 'NO2')
    # BC vs NOX
    make_plot(dataset['BC'], dataset['NOX'], 'BC', 'NOX')
    
def plots_bc_concentration_vs_others():
    print("Plotting BC concentration vs. others...")

def plots_temporal_trends():
    print("Plotting temporal trends...")


if __name__ == "__main__":
    dataset = get_datasets()
    get_stats(dataset)
    normalized_dataset = normalization(dataset)
    get_stats(normalized_dataset)
    show_plots(normalized_dataset)
    correlation_matrix(normalized_dataset)
    scatter_plots(normalized_dataset)

    plots_bc_concentration_vs_others()
    plots_temporal_trends()