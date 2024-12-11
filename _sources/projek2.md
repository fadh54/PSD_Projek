---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Prediksi Penjualan Retai

### Latar Belakang

Di era digital saat ini, pemilik retail harus mampu memahami strategi pemasaran yang efektif. Dengan begitu pemilik retail dapat dengan mudah mengatur kesediaan produk yang akan dipasarkan

### Tujuan

Untuk mengelola peningkatan penjualan toko retail

### Rumusan Masalah

Bagaimana prediksi penjualan 7 hari kedepan?

```{code-cell}
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

```{code-cell}
# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/fadh54/Tugas_PSD/refs/heads/main/Retail.csv"
data = pd.read_csv(url, delimiter=';')
print(data)
```

# Step 2: Data Preprocessing

```{code-cell}

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d/%m/%Y %H:%M')
sales_column_name = 'Quantity'
daily_sales = data.groupby(data['InvoiceDate'].dt.date)[sales_column_name].sum().reset_index()
daily_sales.columns = ['InvoiceDate', 'sales']


for lag in range(1, 8):
    daily_sales[f'lag_{lag}'] = daily_sales['sales'].shift(lag)

# Drop rows with NaN values (due to lagging)
daily_sales = daily_sales.dropna()

# Step 3: Feature Engineering
# Use lag features as predictors
features = [col for col in daily_sales.columns if 'lag' in col]
X = daily_sales[features]
y = daily_sales['sales']

# Split data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Step 4: Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Recursive Multi-step Forecasting
def multi_step_forecast(model, X_last, n_steps):
    predictions = []
    X_current = X_last.copy()

    for _ in range(n_steps):
        y_pred = model.predict([X_current])[0]  # Predict next step
        predictions.append(y_pred)

        # Update features (shift by 1 step, add predicted value as the latest lag)
        X_current = np.roll(X_current, shift=-1)
        X_current[-1] = y_pred

    return predictions

# Use the last row of X_test as the starting point for forecasting
X_last = X_test.iloc[-1].values
n_steps = 7  # Number of days to predict

predictions = multi_step_forecast(model, X_last, n_steps)

# Step 6: Evaluate and Plot Results
print("Predicted sales for the next 7 days:", predictions)

# Plot actual vs predicted sales
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_steps + 1), predictions, marker='o', label='Predicted Sales')
plt.title('Multi-step Sales Forecasting')
plt.xlabel('Days Ahead')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()
```

Kesimpulan dari hasil prediksi menggunakan ke-3 model tersebut, didapatkan best model nya yaitu prediksi menggunakan Linear regression.
