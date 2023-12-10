import torch
from typing import List
import heapq 
from bert4vec import Bert4Vec


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
            self.use_count = []
    # vector 代表存入信息 ， None 代表没有信息
    def add(self,key,value):
        if key.any() == None:
            raise RuntimeError("key == None , memory is not added!")
            return
        elif value.any() == None:
            if key in self.key:
                index = self.key.index(key)
                self.value[index] = None
            else:
                self.key.append(key)
                self.value.append(None)
        else:
                self.key.append(key)
                self.value.append(value)
        self.use_count.append(0)

    def update_usage(self,index):
        if self.usage_count == False:
            return
        elif len(self.use_count) - 1 < index:
            print("index is out of range , nothing changed!!")
        else:
            self.use_count[index] += 1
    def get_usage(self):
        # return normalized usage
        if not self.usage_count:
            raise RuntimeError('I did not count usage!')
        else:
            return self.use_count
    def find_max_usage(self):
        usage = self.get_usage()  
        max_value = max(usage)  
        max_index = usage.index(max_value) 
        return max_index
    def remove_obsolete_features(self, max_size: int):  
        usage = self.get_usage()  
        # 获取最大的n个元素的索引,如果有重复的元素，取前x个
        # 使用堆来存储最大的 n 个元素的索引   
        heap = []  
        # 遍历列表，将元素及其索引加入堆中  
        for i, x in enumerate(usage):  
            heapq.heappush(heap, (x, i))  
        # 取出堆中最大的 n 个元素及其索引  
        top_n = heapq.nlargest(max_size, heap) 
        indices = [index for _, index in top_n] 
        self.use_count = [usage[i] for i in indices] if len(self.use_count) > max_size else self.use_count[:max_size]  
        self.key = [self.key[i] for i in indices] if len(self.key) > max_size else self.key[:max_size]  
        self.value = [self.value[i] for i in indices] if len(self.value) > max_size else self.value[:max_size]

    def engaged(self):
        return self.key is not None

    
    def size(self):
        if self.key is None:
            return 0
        else:
            return len(self.key)

    
    def num_groups(self):
        return len(self.v)

    
    def key(self):
        return self.key


    def value(self):
        return self.value
def model_init(path):
    model = Bert4Vec(model_name_or_path=path)
    return model 

def get_embedding_sentences_bert4vec(model,sentences):
    a_vecs = model.encode(sentences, convert_to_numpy=True, normalize_to_unit=False, batch_size=1)
    embeddings = a_vecs / (a_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return embeddings
if __name__ == "__main__":
    path = '/home/kit/clustering4server_simple/resource/roformer-sim-small-chinese'
    model = model_init(path)
    key_list = ["你喜欢旅游吗？","你曾经去过哪些地方旅行？","听起来很不错！你最喜欢的旅行经历是什么？"]
    value_list = ["我很喜欢旅游。我认为旅行可以让我放松身心，并学习到很多不同的文化。","我曾经去过欧洲和亚洲的一些国家。我很喜欢意大利的威尼斯，那里真的很美。","我的一次旅行经历是在中国的西藏。那里的自然风光和人文景观令人难以忘怀。"]
    work_memory = KeyValueMemoryStorage(True)
    longterm_memory = KeyValueMemoryStorage(False)
    #key (1,384)  value (1,384)
    for i in range(len(key_list)):
        work_memory.add(get_embedding_sentences_bert4vec(model,[key_list[i]]),get_embedding_sentences_bert4vec(model,[value_list[i]]))
    print(type(work_memory.key))
    for i in range(3):
        work_memory.update_usage(1)
    for i in range(2):
        work_memory.update_usage(0)
    for i in range(2):
        work_memory.update_usage(2)
    print(work_memory.get_usage(),len(work_memory.key))
    work_memory.remove_obsolete_features(2)
    print(work_memory.get_usage(),len(work_memory.key))
  

        