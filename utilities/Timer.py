import pickle
import time
from collections import Counter


class Timer:
    def __init__(self, name, save_path):
        self.name = name
        self.counter = dict()
        self.save_path = save_path

    def record(self, index):
        if index not in self.counter.keys():
            self.counter[index] = time.time()

    def get_record(self, index):
        if index not in self.counter.keys():
            return 0
        else:
            return self.counter[index]

    def get_timer(self):
        return self.counter

    def save(self):
        file = open(self.save_path, 'wb')
        pickle.dump(self.counter, file)
        print('Saved the {} timer!'.format(self.name))

    def load(self):
        file = open(self.save_path, 'rb')
        self.counter = pickle.load(file)
        file.close()

    def find_difference(self, timer):
        dict1 = Counter(self.counter)
        dict2 = Counter(timer.get_timer())
        dict1.subtract(dict2)
        for key in dict1.keys():
            dict1[key] = dict1[key] * 1000
        return dict1
