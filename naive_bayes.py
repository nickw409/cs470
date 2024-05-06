import random
import pandas
import numpy

class Naive_Bayes:

   def __init__(self, training_data):
      self.probabilities = {}
      self.prior_prob = 0
      self.num_spam = 0
      self.num_not_spam = 0
      self.training_data = training_data


   def calc_prior_prob(self):
      self.prior_prob = self.num_spam / (self.num_spam + self.num_not_spam)


   def calc_likelihood_prob(self):
      alpha = 1
      num_occur = {}
      columns = self.training_data.columns[:-5]
      num_occur["spam"] = {columns[i]: 0 for i in range(len(columns))}
      num_occur["not_spam"] = {columns[i]: 0 for i in range(len(columns))}
      
      # get num of occurrences of each word in data
      for index, row in self.training_data.iterrows():
         for col in columns:
            if row["spam"] == 1 and row[col] > 0:
               num_occur["spam"][col] = num_occur["spam"][col] + 1
            elif row["spam"] == 0 and row[col] > 0:
               num_occur["not_spam"][col] = num_occur["not_spam"][col] + 1
      
      self.probabilities["spam"] = {columns[i]: 0 for i in range(len(columns))}
      self.probabilities["not_spam"] = {columns[i]: 0 for i in range(len(columns))}
      
      for col in columns:
         #laplace smoothing
         self.probabilities["spam"][col] = (num_occur["spam"][col] + alpha) / (self.num_spam + len(columns) * alpha)
         self.probabilities["not_spam"][col] = (num_occur["not_spam"][col] + alpha) / (self.num_not_spam + len(columns) * alpha)


   def sum_spam(self):
      spam_class = self.training_data['spam']
      for sp in spam_class:
         if sp == 1:
            self.num_spam += 1
         else:
            self.num_not_spam += 1


   def train(self):
      self.sum_spam()
      self.calc_prior_prob()
      self.calc_likelihood_prob()


   def performance(self, test_data):
      correct = 0
      fp = 0
      total_negatives = 0
      tp = 0
      total_positives = 0
      total = 0
      columns = self.training_data.columns[:-5]
      for index, row in test_data.iterrows():
         spam_prob = self.prior_prob
         not_spam_prob = 1 - self.prior_prob
         for col in columns:
            if row[col] > 0:
               spam_prob = spam_prob * self.probabilities["spam"][col]
               not_spam_prob = not_spam_prob * self.probabilities["not_spam"][col]

         if spam_prob > not_spam_prob and row["spam"] == 1:
            correct += 1
            tp += 1
            total_positives = total_positives + 1
         elif spam_prob > not_spam_prob and row["spam"] == 0:
            fp += 1
            total_negatives = total_negatives + 1
         elif spam_prob < not_spam_prob and row["spam"] == 0:
            correct += 1
            total_negatives = total_negatives + 1
         elif spam_prob > not_spam_prob and row["spam"] == 1:
            total_positives = total_positives + 1
         total += 1

      
      accuracy = correct / total
      fp = fp / total_negatives
      tp = tp / total_positives

      print("Accuracy: " + str(accuracy))
      print("FP: " + str(fp))
      print("TP: " + str(tp))


def main():
   filename = "spambase.csv"
   data = pandas.read_csv(filename)

   train_data = data.sample(frac=0.8, random_state=random.randint(1, 10000000))
   test_data = data.drop(train_data.index).sample(frac=1.0, random_state=random.randint(1, 10000000))

   model = Naive_Bayes(train_data)
   model.train()
   model.performance(test_data)


main()