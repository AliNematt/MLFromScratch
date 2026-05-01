import math

class LogisticRegression:
    def __init__(self, learning_rate=0.1):
        self.w = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.mse = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def fit(self, x, y, epoches=100):
        for epoch in range(epoches):
            for i in range(len(x)):
                z = self.w*x[i] + self.b
                y_pred = self.sigmoid(z)
                error = y_pred - y[i]
                self.w -= self.learning_rate * error * x[i]
                self.b -= self.learning_rate * error

        total_error = 0
        for i in range(len(x)):
            z = self.w*x[i] + self.b
            y_pred = self.sigmoid(z)
            total_error += (y_pred - y[i])**2
        self.mse = total_error / len(x)

    def predict(self, x):
        return [self.sigmoid(self.w*xi + self.b) for xi in x]

    
x = [0, 5, 10, 15, 30, 40, 60, 70, 85] 
y = [0, 0, 0, 0, 0, 0, 1, 1, 1]     

model = LogisticRegression(learning_rate=0.1);
model.fit(x, y, epoches=1000)
predictions = model.predict([12, 65, 95, 96])

print(predictions, model.mse)

