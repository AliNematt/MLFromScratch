class KNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def distance(self, a, b):
        return abs(a - b)
    
    def predict_one(self, x_query):
        pairs = []

        for i in range(len(self.x_train)):
            d = self.distance(self.x_train[i], x_query)
            pairs.append((d, self.y_train[i]))

        pairs.sort(key=lambda t: t[0])

        votes = 0
        for j in range(self.k):
            votes += pairs[j][1]

        return 1 if votes >= (self.k / 2) else 0
    
    def predict(self, x_list):
        return [self.predict_one(x) for x in x_list]
    

x_train = [0, 5, 10, 15, 30, 40, 60, 70, 85]
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1]

model = KNN(k=4)
model.fit(x_train, y_train)

prediction = model.predict([10, 35, 65])
print(prediction)
