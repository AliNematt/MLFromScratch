class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0
        self.b = 0
        self.MSE = 0
        self.learning_rate = learning_rate

    def fit(self, x, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(x)):
                y_pred = self.w*x[i] + self.b
                error = y_pred - y[i]
                self.w -= self.learning_rate * (2*error*x[i])
                self.b -= self.learning_rate * (2*error)
        self.MSE = error ** 2

    def predict(self, x):
        return [self.w*xi + self.b for xi in x]

x = [1, 2, 3, 4, 5]
y = [5, 7, 9, 11, 13]

model = LinearRegression(learning_rate=0.1)
model.fit(x, y, epochs=80)
predictions = model.predict(x)

print(predictions, model.MSE)