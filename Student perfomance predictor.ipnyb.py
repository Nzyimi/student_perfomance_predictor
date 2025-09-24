# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data preparation
data = {
    'Hours': [1, 2, 3, 4.5, 5, 6, 7, 8, 9, 10],
    'Scores': [20, 30, 50, 52, 60, 62, 70, 78, 85, 95]
}
df = pd.DataFrame(data)

# Data visualization
plt.figure(figsize=(8, 5))
plt.scatter(df['Hours'], df['Scores'], color='red')
plt.title('Study Hours vs Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.show()

# Model building
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Scores:")
print(comparison)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# User interaction
def predict_score(hours):
    hours_array = np.array(hours).reshape(-1, 1)
    predicted_score = model.predict(hours_array)
    return predicted_score[0]

print("\nScore Prediction Tool")
study_hours = float(input("Enter number of study hours: "))
predicted_score = predict_score(study_hours)
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")