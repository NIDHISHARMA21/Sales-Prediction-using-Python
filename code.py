
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load and preprocess the dataset
data = pd.read_csv(r"E:\4_sales_prediction_project\advertising.csv").astype('float32')

# 2. Check for duplicates
print(f"Number of duplicate rows: {data.duplicated().sum()}")

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(16, 12))
sns.pairplot(data)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt=".2f", center=0, linewidths=0.5)
plt.title('Sales Correlation Heatmap')
plt.show()

data.hist(bins=20, figsize=(10, 8))
plt.show()

# 4. Outlier Detection and Treatment using IQR Method
def treat_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    data[column] = np.clip(data[column], lower_limit, upper_limit)

for column in ['TV', 'Radio', 'Newspaper']:
    treat_outliers(column)

# Recheck for outliers using boxplots
plt.figure(figsize=(12, 8))
for i, column in enumerate(['TV', 'Radio', 'Newspaper', 'Sales'], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data[column])
plt.tight_layout()
plt.show()

# 5. Feature Scaling
scaler = MinMaxScaler()
data[['TV', 'Radio', 'Newspaper']] = scaler.fit_transform(data[['TV', 'Radio', 'Newspaper']])

# 6. Splitting the data into train and test sets
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training and Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print('-' * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred, name)

# 8. Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=models["Random Forest Regressor"], param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)

print_metrics(y_test, y_pred_best_rf, "Best Random Forest Regressor")

# 9. Save the Best Model and Scaler using Pickle
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Best Random Forest Regressor model saved as 'best_rf_model.pkl'")









