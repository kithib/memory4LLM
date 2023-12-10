import torch
from typing import List
from heapq import nlargest
import numpy as np  
from bert4vec import Bert4Vec
from memory_kv_storage import KeyValueMemoryStorage,get_embedding_sentences_bert4vec,model_init

# main function manages three types of memory in a unified manner
# 1.concat long-term memory value/key & work memory value/key
# 2. Similarity calculation (key and query), shape (1,384)
# 3. Find the appropriate value based on the maximum value of similarity
# 4.update memory
class MemoryManager:
    def __init__(self,sensor_memory:list,work_memory:KeyValueMemoryStorage,longterm_memory:KeyValueMemoryStorage):
        self.sensor_memory = sensor_memory
        self.work_memory = work_memory
        self.longterm_memory = longterm_memory
        self.key = []
        self.value = []
        self.value_selected_thisRound = []
        self.max_work_memory = 5
        self.max_longterm_memory = 5

    def concat_two_list(self,listwork,listlong):
        # Convert two lists to numpy arrays 
        arr1 = np.array(listwork) 
        arr2 = np.array(listlong)  
        # Use numpy's concatenate function to connect two arrays 
        arr3 = np.concatenate((arr1, arr2))
        # Convert back to list  
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
    #n can control the selection of the first few
    def find_value(self,query,n:int = 1):
        result = self.find_max_cosine(self.key,query,n)
        index_list = []
        for item in result:
            index_list.append(item[0])
        value_list = self.value[index_list]
        self.value_selected_thisRound = value_list
    def work2longterm(self):
        if len(self.work_memory.key) > self.max_work_memory:
            max_index = self.work_memory.find_max_usage()
            self.longterm_memory.add(self.work_memory.key[max_index],self.work_memory.value[max_index])
            del self.work_memory.key[max_index]  
            del self.work_memory.value[max_index]  
    #Set workmemoey to be updated every 3 times
    def update_memory(self,answer,count,new_key,new_value):

        #sensor memory update
        model = model_init('/home/kit/clustering4server_simple/resource/roformer-sim-small-chinese')
        embedding_answer = get_embedding_sentences_bert4vec(model,answer)
        generate_feature = sum(self.value_selected_thisRound) / len(self.value_selected_thisRound)
        self.sensor_memory = (self.sensor_memory + embedding_answer + generate_feature) / 3

        #work memory and long term memory update
        if count % 3 == 0:
            self.work_memory.add(new_key,new_value)
            self.work2longterm()
            self.longterm_memory.remove_obsolete_features(self.max_longterm_memory)

if __name__ == "__main__":
    memorymanger = MemoryManager()

        

