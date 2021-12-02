#%%

import numpy as np
import matplotlib.pyplot as plt
from prml.preprocess import PolynomiaFeature
from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianRegression
)
#%%
np.random.seed(1234)

def create_dummy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std,
                                   size=x.shape)
    return x, t


def data_pattern(x):
    return np.sin(2 * np.pi * x)

x_train, y_train = create_dummy_data(data_pattern,
                                     5, 0.5)

x_test = np.linspace(0, 1, 100)
y_test = data_pattern(x_test)

#%%

#degree 0
#%%
degree = 0
feature = PolynomialFeature(degree)
x_train_0 = feature.transform(x_train)
x_test_0 = feature.transform(x_test)
model = LinearRegression()
model.fit(x_train_0, y_train)
y_0 = model.predict(x_test_0)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_0, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%%

#degree 1
#%%
degree = 1
feature = PolynomialFeature(degree)
x_train_1 = feature.transform(x_train)
x_test_1 = feature.transform(x_test)
model = LinearRegression()
model.fit(x_train_1, y_train)
y_1 = model.predict(x_test_1)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_0, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%%

#degree 2
#%%
degree = 2
feature = PolynomialFeature(degree)
x_train_2 = feature.transform(x_train)
x_test_2 = feature.transform(x_test)
model = LinearRegression()
model.fit(x_train_2, y_train)
y_2 = model.predict(x_test_2)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_2, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%%

#degree 3
#%%
degree = 3
feature = PolynomialFeature(degree)
x_train_3 = feature.transform(x_train)
x_test_3 = feature.transform(x_test)
model = LinearRegression()
model.fit(x_train_3, y_train)
y_3 = model.predict(x_test_0)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_3, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%%

#degree 4
#%%
degree = 4
feature = PolynomialFeature(degree)
x_train_4 = feature.transform(x_train)
x_test_4 = feature.transform(x_test)
model = LinearRegression()
model.fit(x_train_4, y_train)
y_4 = model.predict(x_test_4)
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

plt.plot(x_test, y_0, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%%

#show plots from degree 0 to 3 in one figure
#%%

#%%

#RMSE
#%%
def rmse(predicted_val, true_val):
    return np.sqrt(
        np.mean(
            np.square(
                predicted_val - true_val
                )
            )
        )

rmse_2 = rmse(y_2, y_test)
rmse_3 = rmse(y_3, y_test)
rmse_4 = rmse(y_4, y_test)
print("The RMSE of a model of degree 2 is", 
      rmse_2)
print("The RMSE of a model of degree 3 is", 
      rmse_3)
print("The RMSE of a model of degree 4 is", 
      rmse_4)
#%%


#%%

#%%