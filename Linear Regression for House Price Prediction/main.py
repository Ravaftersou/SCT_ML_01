import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")

x = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("========== Model Evaluation ==========")
print(f"Mean Squared Error (MSE) : {mean_squared_error(y_test, y_pred): .2f}")
print(f"R square Score {r2_score(y_test, y_pred): .2f}")

try:
    sqft = float(input("Enter the square footage of the house: "))
    beds = int(input("Enter the number of bedrooms: "))
    baths = int(input("Enter the no. of bathrooms: "))

    user_input = np.array([[sqft, beds, baths]])
    predicted_score = model.predict(user_input)

    print(f"Predicted House Price: Rs. {predicted_score[0]:,.2f} ")

except Exception as e:
    print("Invalid input:", e)


print("\n========= Model Coefficents =========")
print(f"Imtercept: {model.intercept_:.2f}")
for feature, coef in zip(x.columns, model.coef_):
    print(f" - {feature}: {coef:.2f}")
