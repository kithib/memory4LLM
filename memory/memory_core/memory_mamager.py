import torch
from typing import List
from heapq import nlargest
import numpy as np  
from bert4vec import Bert4Vec
from memory_kv_storage import KeyValueMemoryStorage

# main function  统一管理三种记忆
# 1.concat long-term memory value/key  &  work memory value/key
# 2.相似度计算 （key 和 query ）,shape (1,384)
# 3.根据相似度的最大值找到合适的value
# 4.更新敏感记忆（sensor memory）
class MemoryManager:
    def __init__(self,sensor_memory:list,work_memory:KeyValueMemoryStorage,longterm_memory:KeyValueMemoryStorage):
        self.sensor_memory = sensor_memory
        self.work_memory = work_memory
        self.longterm_memory = longterm_memory
        self.key = []
        self.value = []

    def concat_two_list(self,listwork,listlong):
        # 将两个列表转换为numpy数组 
        arr1 = np.array(listwork) 
        arr2 = np.array(listlong)  
        # 使用numpy的concatenate函数连接两个数组  
        arr3 = np.concatenate((arr1, arr2))
        # 转换回列表（如果需要）  
        listall = arr3.tolist()  
        return listall
    

    def concat_work_longterm_memory(self):
        self.key = self.concat_two_list(self.work_memory.key,self.longterm_memory.key)
        self.value = self.concat_two_list(self.work_memory.value,self.longterm_memory.value)

    def find_max_cosine(vectors, x , n):
        cosine_similarities = [np.dot(x, vector) / (np.linalg.norm(x) * np.linalg.norm(vector)) for vector in vectors]
        max_indices = nlargest(n, range(len(cosine_similarities)), key=cosine_similarities.__getitem__)
        result = []
        for index in max_indices:
            result.append((index, cosine_similarities[index]))
        return result 
    #n 可以控制选前几个
    def find_value(self,query,n:int = 1):
        result = self.find_max_cosine(self.key,query,n)
        index_list = []
        for item in result:
            index_list.append(item[0])
        value_list = self.value[index_list]
        return value_list
    


        

