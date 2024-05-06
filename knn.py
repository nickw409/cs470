import random
import pandas
import multiprocessing

class KNN:

   def __init__(self, K, training_data, testing_data):
      self.K = K
      self.training_data = training_data
      self.testing_data = testing_data

   
   #Using euclidean distance
   def calc_distance(self, test_row):
      total = 0
      columns = self.training_data.columns[:-1]
      distances = []
      
      for train_row in self.training_data.to_records():
         total = 0
         for col in columns:
            total += (test_row[col] - train_row[col])**2
         total = total**0.5
         distances.append((train_row["spam"], total))
      
      distances.sort(key=lambda a: a[1])
      return distances


   def is_spam(self, test_row):
      num_spam = 0
      num_not_spam = 0
      selected_distances = self.calc_distance(test_row)
      for i in range(self.K):
         result, distance = selected_distances[i]
         if result == 1:
            num_spam += 1
         else:
            num_not_spam += 1
      
      if num_spam > num_not_spam and test_row["spam"] == 1:
         return True, True
      elif num_spam > num_not_spam and test_row["spam"] == 0:
         return True, False
      elif num_spam < num_not_spam and test_row["spam"] == 1:
         return False, True
      else:
         return False, False
   

   def find_optimal_K(self):
      accuracies = []
      
      for self.K in range(1, 25, 2):
         accuracy = self.performance()
         accuracies.append((self.K, accuracy))
         print("Accuracy with K value: " + str(self.K))
      
      accuracies.sort(reverse=True, key=lambda a: a[1])
      return accuracies[0]


   def performance(self):
      print("Computing KNN...")
      correct = 0
      fp = 0
      total_negatives = 0
      tp = 0
      total_positives = 0
      total = 0
      count = 0

      pool = multiprocessing.Pool()
      results_async = [pool.apply_async(self.is_spam, args = (row, )) for row in 
                        self.testing_data.to_records()]
      results = [r.get() for r in results_async]

      for result in results:
         if result[0] and result[1]:
            correct += 1
            tp += 1
            total_positives = total_positives + 1
         elif result[0] and not result[1]:
            fp += 1
            total_negatives = total_negatives + 1
         elif not result[0] and not result[1]:
            correct += 1
            total_negatives = total_negatives + 1
         elif not result[0] and result[1]:
            total_positives = total_positives + 1
         total += 1

      accuracy = correct / total
      fp = fp / total_negatives
      tp = tp / total_positives

      print("Accuracy: " + str(accuracy))
      print("FP: " + str(fp))
      print("TP: " + str(tp))

      return accuracy


def main():
   filename = "spambase.csv"
   data = pandas.read_csv(filename)

   train_data = data.sample(frac=0.5, random_state=random.randint(1, 10000000))
   test_data = data.drop(train_data.index).sample(frac=0.1, random_state=random.randint(1, 10000000))
   K = 1

   model = KNN(K, train_data, test_data)
   
   optimal_K = model.find_optimal_K()
   print(optimal_K)

   model.K = optimal_K[0]
   model.training_data = data.sample(frac=0.8, 
                                    random_state=random.randint(1, 10000000))
   model.testing_data = data.drop(train_data.index).sample(frac=1.0, 
                                    random_state=random.randint(1, 10000000))
   model.performance()


main()