# House Price Prediction Model

This script demonstrates how to build and evaluate a linear regression model to predict house prices based on three features: size, number of rooms, and age of the house.

## Requirements

Before running the script, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset used in this script is assumed to be a CSV file located at `../houses-data/datos.csv`. The dataset should include the following columns:

- **Tamaño**: Size of the house
- **Num_Habitaciones**: Number of rooms
- **Edad**: Age of the house
- **Precio**: Price of the house (target variable)

## Script Overview

1. **Data Reading**: The script reads the house data from the CSV file.

    ```python
    read_csv = pd.read_csv("../houses-data/datos.csv")
    ```

2. **Feature Selection**: The script selects three features (`Tamaño`, `Num_Habitaciones`, `Edad`) to predict the house price (`Precio`).

    ```python
    x = read_csv[['Tamaño','Num_Habitaciones' ,'Edad']]
    y = read_csv['Precio']
    ```

3. **Data Splitting**: The data is split into training and testing sets (80% training, 20% testing).

    ```python
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    ```

4. **Model Training**: A linear regression model is created and trained on the training data.

    ```python
    model = LinearRegression()
    model.fit(x_train, y_train)
    ```

5. **Model Prediction**: The trained model is used to predict house prices on the test data.

    ```python
    y_pred = model.predict(x_test)
    ```

6. **Performance Evaluation**: The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.

    ```python
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ```

7. **Manual Prediction**: The script also includes an example of manual predictions and their evaluation.

    ```python
    y_test_manual = np.array([300, 400, 500, 600, 700])
    y_pred_manual = np.array([320, 380, 520, 590, 720])
    mse_manual = mean_squared_error(y_test_manual, y_pred_manual)
    r2_manual = r2_score(y_test_manual, y_pred_manual)
    ```

## Output

The script prints the following metrics to evaluate the model's performance:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values.

    ```python
    print('Squared error of performance (MSE):', mse)
    ```

2. **R-squared (R²)**: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.

    ```python
    print('Print performance metrics (R²):', r2)
    ```

3. **Manual MSE and R²**: Provides a comparison of manual predictions versus actual values.

    ```python
    print('Coefficient of Manual Determination (R²):', r2_manual)
    print('Manual Mean Square Error:', mse_manual)
    ```
