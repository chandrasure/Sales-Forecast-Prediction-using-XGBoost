# Sales-Forecast-Prediction-using-XGBoost

**Project Description**
A time series sales forecasting model built using Python and XGBoost. This project uses historical sales data to predict future sales by capturing temporal patterns through lagged features. It helps businesses optimize inventory, marketing, and demand planning with data-driven decision-making.

**README.md**
markdown
Copy
Edit
#  Sales Forecast Prediction using XGBoost

This project demonstrates how to build a **sales forecasting model** using Python and **XGBoost**, a powerful gradient boosting algorithm. The model is trained on historical sales data and uses lag-based feature engineering to forecast future sales trends.

---

##  Objective

Sales forecasting is crucial for making informed decisions about **inventory management**, **marketing**, and **resource planning**. This project aims to predict future sales using past sales data and visualize the forecast.

---

##  Dataset

The dataset contains features such as:

- Row ID
- Order ID
- Order Date
- Customer ID
- Sales

Ensure the CSV file (e.g., `train.csv`) is placed in the working directory.

---

##  Libraries Used

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
pandas – data manipulation

matplotlib / seaborn – data visualization

scikit-learn – model evaluation and splitting

xgboost – regression model

**Workflow**
1. Import Libraries
python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
2. Load & Preprocess Data
Parse Order Date

Aggregate daily sales

python
Copy
Edit
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')
sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()
3. Visualize Sales Trend
python
Copy
Edit
plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'])
4. Feature Engineering – Lag Features
python
Copy
Edit
def create_lagged_features(data, lag=5):
    for i in range(1, lag+1):
        data[f'lag_{i}'] = data['Sales'].shift(i)
    return data.dropna()
5. Train-Test Split
python
Copy
Edit
X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
y = sales_with_lags['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
6. Train XGBoost Regressor
python
Copy
Edit
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
7. Predict & Evaluate
python
Copy
Edit
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.2f}")
8. Visualize Predictions
python
Copy
Edit
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')

**Results**
RMSE: ~734.63

The prediction curve closely follows the actual trend, indicating good performance.

 **Future Improvements**
Incorporate holiday and promo effects as external regressors.

Try advanced models like LSTM or Prophet.

Deploy using Flask/Django for business use.

 **Conclusion**
This project highlights how machine learning, particularly XGBoost with lag-based features, can be leveraged for reliable and scalable sales prediction. Such models help businesses make smarter, data-driven decisions.
