import torch
from typing import List

class KeyValueMemoryStorage:
    """
    this class works for Key Value form memory storage
    working memory and long-term memory
    """
    """
    The key is the embedding of the prompt and is stored in a list (vector in it).
    The value is the embedding of the answer and is stored in a list (vector in it).
    KeyValueMemoryStore functions include:
    1.Initialization information
    2. add memory
    3. Calculate the number of uses (1. The basis for whether working memory is added to long-term memory, 2. The basis for forgetting long-term memory)
    4. Delete unnecessary long-term memory
    5. write imformation to csv 
    """
    #usage_count true:count memory usage , false:not count memory usage
    def __init__(self,usage_count:bool):
        self.usage_count = usage_count

        self.key = [] # vector in it
        self.value = []

        if usage_count:
            self.use_count = 0
    # vector 代表存入信息 ， None 代表没有信息
    def add(self,key,value):
        if key == None:
            print("key == None , memory is not added!")
        elif value == None:
            if key in self.key:
                index = self.key.index(key)
                self.value[index] = None
            else:
                self.key.append(key)
                self.value.append(None)
        else:
            if key in self.key:
                index = self.key.index(key)
                self.value[index] = value
            else:
                self.key.append(key)
                self.value.append(value)

        