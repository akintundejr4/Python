import os 
import sys
import ssl
import time
import json

class Tools: 
    """ Helper methods for an experimentation class.""" 

    def parse_dict(self, my_dict, result):
        for val in my_dict.items():
            if type(val) == list:
                for i in val:
                    result.append(i)
            elif type(val) == dict:
                self.parse_dict(val, result)
            else:
                result.append(val)
        return result

    def yield_stuff(self): 
        for i in range(10): 
            yield i * i 

    def quicksort(self, listToSort): 
        leftList = []
        rightList = []
        lastNumber = listToSort[len(listToSort) - 1]

        for number in listToSort: 
            if(number <= lastNumber): leftList.append(number)
            if(number > lastNumber): rightList.append(number)

        return self.quicksort(leftList + rightList)


tools = Tools() 
theArray = [3, 9, 4, 8, 1, 10, 0, 5, 7, 2, 6]

my_dict0 = {'root': {'b1': [1],
                     'b2': [2]}
           }

my_dict1 = {'root': {'b1': {'leaf1': [1,2,3],
                            'leaf2': [4],
                            'leaf3': [5,6]
                           },
                     'b2': {'leaf1': [7,8],
                            'leaf2': [9,10,11]
                           },
                     'b3': 12
                    }
           }

result = []
# tools.parse_dict(my_dict1, result)

myGen = tools.yield_stuff() 


# print(quicksort(theArray))
