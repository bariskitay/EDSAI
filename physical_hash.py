#!/usr/bin/env python
# coding: utf-8

# ## Exercise 4 (4 Points)
# 
# In the lecture, you have learned about the concept of logical and physical indexes. In this exercise, your task is to implement a very simple _physical_ hash index.
# For this, we provide you with an incomplete `HashIndex` class which implements the constructor `__init__(self, num_buckets)` and a method to visualize the current index state `show_index(self)`. Assume that we exclusively want to index elements of type `int`. The `HashIndex` contains the following instance variables:
# 
# * `self.num_buckets`, integer representing the number of hash buckets
# * `self.buckets`, list of lists containing the indexed items
# 
# 
# You have to implement the following methods of the `HashIndex` class:
# 
# * `insert(self, item)`
#     * use the identity function in combination with `modulo` to compute the corresponding bucket for the given `item`
#     * the buckets should contain the elements in insertion order
#     
#     
# * `lookup(self, item)`
#     * search for the given `item`
#     * return the number of required indirections to find `item`, i.e. the number of required comparisons
#     * returns $-1$ if `item` does not exist

# In[2]:


class HashIndex:
    def __init__(self, num_buckets=4): 
        self.num_buckets = num_buckets
        self.buckets = []
        for i in range (0, num_buckets):
            self.buckets.append([])
        
    # visualize the contents of this index:
    def show_index(self):
        for i in range(self.num_buckets):
            print("bucket " +  str(i) + ": ", end="")
            for elem in self.buckets[i]:
                print("➜ " + str(elem), end=" ")
            print()
            
    # insert the element `item` into the index    
    def insert(self, item):
        assert(isinstance(item, int)), "function argument has to be of type int"
        hashed_val = hash(item)
        index = hashed_val % 4
        self.buckets[index].append(item)
        
    # search for the element `item` and return the number of required indirections
    def lookup(self, item):
        assert(isinstance(item, int)), "function argument has to be of type int"
        hashed_val = hash(item)
        index = hashed_val % 4
        bucket = self.buckets[index]
        c = 0
        for key in bucket :
            c += 1
            if key == item:
                return c
        return -1

