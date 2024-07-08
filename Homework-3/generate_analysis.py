import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import math

name = "KR"

df = pd.read_csv(f"final_results_{name}-r2-rmse.csv", sep="; ")

path = f"analysis/{name}/"

# substitute all true values with 1 and false values with 0
df = df.replace("True", 1)
df = df.replace("False", 0)
df = df.replace("scale", 1)
df = df.replace("auto", 2)
df = df.replace("linear", 0)
# df = df.replace("poly", 1)
df = df.replace("rbf", 1)
df = df.replace("sigmoid", 2)
df = df.replace(True, 1)
df = df.replace(False, 0)
df = df.replace("None", 0)
df = df.replace("uniform", 0)
df = df.replace("distance", 1)
df = df.replace("auto", 0)
# df = df.replace("ball_tree", 1)
# df = df.replace("kd_tree", 2)
# df = df.replace("brute", 3)
# remove a column 
df = df.drop(['model'], axis=1)

print(df.head())

# 1. Correlation Analysis
correlation_matrix = df.corr("spearman")
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(f"{path}correlation.png")
plt.close()

# Display correlation of features with r2 and rmse
print(correlation_matrix[['r2', 'rmse']])

#------------------------------------------------------------
# 2. Feature Importance using RandomForestRegressor
X = df.drop(['r2', 'rmse'], axis=1)
y_r2 = df['r2']
y_rmse = df['rmse']

# Split data into training and testing sets
X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(X, y_r2, test_size=0.2, random_state=42)
X_train_rmse, X_test_rmse, y_train_rmse, y_test_rmse = train_test_split(X, y_rmse, test_size=0.2, random_state=42)

# RandomForestRegressor for r2
rf_r2 = RandomForestRegressor(random_state=42)
rf_r2.fit(X_train_r2, y_train_r2)
feature_importances_r2 = rf_r2.feature_importances_

# RandomForestRegressor for rmse
rf_rmse = RandomForestRegressor(random_state=42)
rf_rmse.fit(X_train_rmse, y_train_rmse)
feature_importances_rmse = rf_rmse.feature_importances_

# Plot feature importances for r2
feat_importances_r2 = pd.Series(feature_importances_r2, index=X.columns)
feat_importances_r2.nlargest(10).plot(kind='barh')
plt.title('Feature Importances for r2')
plt.savefig(f"{path}feature_importances_r2.png")
plt.close()

# Plot feature importances for rmse
feat_importances_rmse = pd.Series(feature_importances_rmse, index=X.columns)
feat_importances_rmse.nlargest(10).plot(kind='barh')
plt.title('Feature Importances for rmse')
plt.savefig(f"{path}feature_importances_rmse.png")
plt.close()

# 3. Feature Importance using LassoCV
lasso_r2 = LassoCV(cv=5, random_state=42).fit(X_train_r2, y_train_r2)
lasso_rmse = LassoCV(cv=5, random_state=42).fit(X_train_rmse, y_train_rmse)

# Display coefficients
lasso_coef_r2 = pd.Series(lasso_r2.coef_, index=X.columns)
lasso_coef_rmse = pd.Series(lasso_rmse.coef_, index=X.columns)

print("Lasso Coefficients for r2:\n", lasso_coef_r2)
print("Lasso Coefficients for rmse:\n", lasso_coef_rmse)

# Plot Lasso coefficients for r2
lasso_coef_r2[lasso_coef_r2 != 0].sort_values().plot(kind='barh')
plt.title('Lasso Coefficients for r2')
plt.savefig(f"{path}lasso_coef_r2.png")
plt.close()

# Plot Lasso coefficients for rmse
lasso_coef_rmse[lasso_coef_rmse != 0].sort_values().plot(kind='barh')
plt.title('Lasso Coefficients for rmse')
plt.savefig(f"{path}lasso_coef_rmse.png")
plt.close()