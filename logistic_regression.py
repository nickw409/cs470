import random
import pandas
import numpy as np

class LR:

   def __init__(self, learning_rate, epochs):
      self.learning_rate = learning_rate
      self.epochs = epochs
   

   def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))
   
   def loss_function(self, y, pred_y):
      n = len(y)
      loss = - np.sum(np.dot(y, np.log(pred_y)) + np.dot((1 - y), 
                                                         np.log(1 - pred_y)))
      loss /= n
      return loss
   

   def fit(self, train_data_X, train_data_Y, val_data_X, val_data_Y):
      self.train_data_X = train_data_X
      self.train_data_Y = train_data_Y

      self.train_data_X1 = np.c_[np.ones((len(train_data_X), 1)), train_data_X]
      val_data_X = np.c_[np.ones((len(val_data_X), 1)), val_data_X]
      self.M = np.random.randn(len(self.train_data_X[0]) + 1, 1)
      
      best_model = self.M
      best_performance = 0
      for i in range(self.epochs):
         y = self.sigmoid(self.M.T.dot(self.train_data_X1.T))
         pred_y = self.sigmoid(self.M.T.dot(self.train_data_X1.T))
         gm = self.train_data_X1.T.dot(pred_y.T - y.T) * (2 / len(y))
         _p = self.performance(val_data_X, val_data_Y)

         if _p > best_performance:
            best_model = self.M
            best_performance = _p
         
         self.M = self.M - self.learning_rate * gm



   def performance(self, val_data_X, val_data_Y):
      pred_y = self.sigmoid(self.M.T.dot(val_data_X.T))
      if pred_y > 0.5:
         return 1
      else:
         return 0
      



def main():
   filename = "spambase.csv"
   data = pandas.read_csv(filename)
   train_data = data.sample(frac=0.8, random_state=random.randint(1, 10000000))
   test_data = data.drop(train_data.index).sample(frac=1.0, random_state=random.randint(1, 10000000))
   val_data = train_data.sample(frac=0.2, random_state=random.randint(1, 10000000))
   
   train_data_X = train_data.iloc[:,:-1].to_numpy()
   train_data_Y = train_data.iloc[:,-1:].to_numpy()

   val_data_X = val_data.iloc[:,:-1].to_numpy()
   val_data_Y = val_data.iloc[:,-1:].to_numpy()

   test_data_X = test_data.iloc[:,:-1].to_numpy()
   test_data_Y = test_data.iloc[:,-1:].to_numpy()

   model = LR(0.01, 50)
   model.fit(train_data_X, train_data_Y, val_data_X, val_data_Y)


main()