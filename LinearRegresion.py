import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("linear.csv")
    print("Data loaded:\n",data.head())
    data.plot.scatter("x","y")
    model = LinearRegression(fit_intercept=True)
    model.fit(data[["x"]],data["y"])
    
    x_fit = pd.DataFrame([data["x"].min(), data["x"].max()])
    y_pred = model.predict(x_fit)
    
    fig, ax = plt.subplots()
    data.plot.scatter("x", "y", ax=ax)
    ax.plot(x_fit[0], y_pred, linestyle=":")
    
    print("\nModel gradient: ", model.coef_[0])
    print("Model intercept: ", model.intercept_)