import os 
import sys
import ssl
import time
import json


theArray = [3, 9, 4, 8, 1, 10, 0, 5, 7, 2, 6]

def quicksort(listToSort): 
    leftList = []
    rightList = []
    lastNumber = listToSort[len(listToSort) - 1]

    for number in listToSort: 
        if(number <= lastNumber): leftList.append(number)
        if(number > lastNumber): rightList.append(number)

    return quicksort(leftList + rightList)

print(quicksort(theArray))
