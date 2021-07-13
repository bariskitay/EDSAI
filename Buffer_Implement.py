#!/usr/bin/env python
# coding: utf-8

# ## Exercise 4 (4 Points)
# 
# In the lecture, you were shown a `Buffer` implementation using **Least Recently Used (LRU)** as replacement strategy. In this exercise, your task is to implement a similar buffer using **Least Frequently Used (LFU)** as replacement strategy. For this, we provide you with an incomplete `LFUBuffer` class which only implements the constructor `__init__(self, maxBufferSize)` and a method to visualize the current buffer state `showBuffer(self)`. Furthermore, the `LFUBuffer` contains the following instance variables:
# 
# * `self.maxBufferSize`, integer representing the maximal amount of buffer slots
# * `self.buffer`, list containing the items (e.g. pageIDs) to buffer
# * `self.frequencies`, dictionary mapping buffer items (e.g. pageIDs) to their reference count
# 
# 
# You have to implement the following methods of the `LFUBuffer` class:
# 
# * `load(self, item)`
#     * append `item` to the **beginning** of the buffer
#     * add a corresponding key-value pair with the initial reference count to the frequency dictionary
#     * if the buffer is full, trigger an `evict()`
#     * Note: you may assume, that the items in the buffer are unique at all times
#     
#     
# * `evict(self)`
#     * identify the item with the lowest reference count
#     * remove the corresponding item form the buffer
#     * remove the corresponding key-value pair from the dictionary
#     * Note: if two or more items have been accessed equally often, the older item in the buffer is evicted
#     
#     
# * `get(self, pos)`
#     * increment the reference count of the item at position `pos` in the buffer

# In[2]:


class LFUBuffer:
    def __init__(self, maxBufferSize): 
        self.maxBufferSize = maxBufferSize 
        self.buffer = []
        self.frequencies = {}
        
    # visualize the contents of this buffer:
    def showBuffer(self):
        print('\t[',end='')
        for i, item in enumerate(self.buffer):
            if i:  # print a separator if this isn't the first element
                print(',', end=" ")
            print("(" + str(item) + ", " + str(self.frequencies[item]) + ")", end="")
        print(']')
        
    # load a data item to *the beginning* of this buffer:
    def load(self, item):
        print("load("+ str(item) +")")
        
        if item in self.buffer :
            self.frequencies[item] += 1 
            index = self.buffer.index(item)
            self.buffer = [self.buffer[index]] + self.buffer[:index] + self.buffer[index+1:]
        else :        
            self.frequencies[item] = 0
            self.buffer = [item] + self.buffer 
        
        if len(self.buffer) > self.maxBufferSize :
            self.evict()
        
        self.showBuffer()
            
    # evict (remove) the item with the smallest count from the buffer:
    def evict(self):  
        print("evict...")
        
        list_counts = list(self.frequencies.values())
        
        if list_counts[-1] <= list_counts[0]:
            if list_counts[-1] <= list_counts[1]:
                lowestRef = list_counts[-1]
            else :
                lowestRef = list_counts[1]
        elif list_counts[1] <= list_counts[0]:
            lowestRef = list_counts[1]
        else : 
            lowestRef = list_counts[0]
        
        for key,value in self.frequencies.items():
            if (value == lowestRef):
                a = key
                self.buffer.remove(a)
                break
        del self.frequencies[a]
        
        self.showBuffer()
        
    # get a handle to a particular item contained in the buffer:
    def get(self, pos):
        print("get("+str(pos) +")" )
        if pos >= len(self.buffer):
            raise Exception("Index out of bounds")
        self.frequencies[self.buffer[pos]] += 1
        self.showBuffer()
        

