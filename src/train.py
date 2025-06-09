from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(X, y):
    """Train a multi-output regression model."""
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with MSE."""
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"MSE: {mse:.4f}")