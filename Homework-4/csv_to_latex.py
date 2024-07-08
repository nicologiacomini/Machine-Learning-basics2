import pandas as pd

df = pd.read_csv('results/unvariate/window-10/incomplete_data_matrix-0.1-5.csv-T10.csv')
df.drop(columns=['poly_rmse', 'poly_r2'], inplace=True)

latex_table = df.to_latex(
    index=False,
    caption='RMSE and R2 values for different sensors, window size = 10',
    label='tab:rmse-r2',
    position='htbp!',
    column_format='|c|c|c|c|c|c|c|',
    escape=False,
    float_format="{:0.2f}".format
)

print(latex_table)