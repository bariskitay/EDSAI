#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# define the unittests
get_ipython().magic('run -i assignment01_unittests')


# ## Exercise 1 (4 Points)

# ### a) Minimum of three numbers (2 Points)
# Implement a function `min_triple(x, y, z)` that returns the smallest number of the three parameters $x$, $y$, and $z$. Do not use the built-in function `min`.

# In[ ]:


def min_triple(x, y, z):
    if x > y : 
        if y > z :
            return z
        else :
            return y
    elif x > z :
        return z 
    else :
        return x


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'MinTripleTest'], verbosity=2, exit=False)


# ### b) Sum of the digits (2 Points)
# Implement a function `sum_digits(x)` that returns the sum of the digits of a non-negative integer $x$. For example:
# 
# $$ \text{sum_digits(1374)} = 1 + 3 + 7 + 4 = 15 $$
# 
# **Hint:** Make clever use of `div` (Python operator `//`) and `modulo` (Python operator `%`) to extract the individual digits. 

# In[ ]:


def sum_digits(x):

    sum = 0

    while x // 10 != 0:
        sum += x % 10
        x //= 10

    sum += x

    return sum


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'SumDigitsTest'], verbosity=2, exit=False)


# ## Exercise 2 (4 Points)
# 
# ### a) Lists (2 Points)
# 1. Implement a function `create_list(low, high, step)` that returns a list containing the integers from _low_ to _high_ (inclusive) with a step size of _step_. For example:
# 
# $$ \text{create_list(0, 10, 3)} = [0, 3, 6, 9] $$
# 
# Your implementation should use a `while` loop. You must not use Python's built-in `range` function. In case _low_ is an integer larger than _high_, your implementation should return an empty list. You may assume that _step_ is a non-negative integer with _step_ $> 0$.

# In[ ]:


def create_list(low,high,step):
    list = []
    while low <= high:
        list.append(low)
        low += step
    
    return list


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'CreateListTest'], verbosity=2, exit=False)


# 2. Implement a function `list_sub_add(input_list)`that returns the sum of alternatingly subtracting and adding the values of _input\_list_. Your implementation should start with subtraction. For example:
# 
# $$ \text{list_add_sub([5, 6, 7, 8, 9, 10])} = (0 -)\ 5 + 6 - 7 + 8 - 9 + 10 = 3 $$

# In[ ]:


def list_sub_add(list):
    res = 0
    for i in list:
        if list.index(i) % 2 == 0 :
            res -= i 
        else :
            res += i
    
    return res


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'ListSubAddTest'], verbosity=2, exit=False)


# ### b) Dictionaries (1 Points)
# 
# Implement a function `create_exp_dict(n,exponent)` that returns a dictionary containing the numbers from $0$ to $n$ (including) as keys and maps them to their corresponding power value:
# 
# $$ \text{base}^\text{exponent} = \text{power},$$
# where _base_ $\in [0, n]$.

# In[ ]:


def create_exp_dict(n,exponent):
    dict = {}
    for i in range(n+1):
        dict[i]=i**exponent
    
    return dict


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'CreateExpDictTest'], verbosity=2, exit=False)


# ### c) Sets (1 Points)
# Implement a function `create_even_set(low, high)` that returns a set containing all **even** numbers from _low_ to _high_ (inclusive). In addition, implement a function `create_odd_set(low, high)` that returns a set containing all **odd** numbers from _low_ to _high_ (inclusive). However, this function should make use of `create_even_set` internally (**Hint:** Make use of set operations).

# In[ ]:


def create_even_set(low,high):
    even_Set = set()
    for i in range(low,high+1):
        if (i % 2 == 0) :
            even_Set.add(i)
            
    return even_Set

def create_odd_set(low,high):
    all_Set = set(range(low,high+1))
    return all_Set.difference(create_even_set(low,high))


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'SetsTest'], verbosity=2, exit=False)


# ## Exercise 3 (8 Points)
# 
# ### a) Median (2 Points)
# Implement a function `median(input_list)` that returns the median of a given _input\_list_. The median is the value separating the lower half from the upper half of a dataset and is defined as follows:
# 
# $$
# median([a_{0}, \cdots, a_{n-1}]) =
# \begin{cases}
# a_{\frac{n-1}{2}}, & n \text{ is odd}\\
# \frac{1}{2} (a_{\frac{n}{2}-1} + a_{\frac{n}{2}})  & n \text{ is even}\\
# \end{cases}
# $$
# 
# where $[a_{0}, \cdots, a_{n-1}]$ is a sorted list of integers and $n$ is the number of elements in the list.
# 
# You may assume that _input\_list_ is not empty but not necessarily sorted. 

# In[ ]:


def sort_list(x):
    for k in range(len(x)-1):
        for i in range(0,len(x)-1-k):
            if x[i]>x[i+1]:
                x[i],x[i+1]=x[i+1],x[i]
    print(x)
    return x

def median(input_list):
    sortedList = sort_list(input_list)
    n = len(input_list)
    if n % 2 != 0:
        return sortedList[int((n-1)/2)]
    else:
        return 0.5*(sortedList[int(n/2-1)]+sortedList[int(n/2)])


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'MedianTest'], verbosity=2, exit=False)


# ### b) Binomial Coefficient (3 Points)
# Implement a function `binomial_coefficient(n,k)` that returns the binomial coefficient of $n$ over $k$ for non-negative integers $n$ and $k$ with $n \geq k$:
# 
# $$ {n \choose k} = \frac{n!}{k! \times (n - k)!} $$
# 
# For this purpose, implement a function `factorial(n)` that returns the factorial of $n$ for $n \geq 1$:
# 
# $$ n! = n \times (n - 1) \times (n - 2) \times ... \times 3 \times 2 \times 1 = \prod_{i=1}^{n}i$$
# 
# The factorial of 0 is 1:
# 
# $$ 0! = 1 $$
# 
# You may assume that the provided numbers are valid.

# In[ ]:


def factorial(n):
    res = 1
    if n == 0 : 
        return 1
    for i in list(range(1,n+1)) :
        res *= i
    return res

def binomial_coefficient(n,k):
    return (factorial(n))/(factorial(k)*factorial(n-k))


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'BinomialCoefficientTest'], verbosity=2, exit=False)


# ### c) Greatest common divisor (3 Points)
# Implement a function `gcd(x, y)` that returns the greatest common divisor of two non-negative integers $x > 0$ and $y >= 0$:
# 
# $$ gcd(x, y) = 
# \begin{cases}
# x, & y = 0\\
# gcd(y, remainder(x, y)), & y > 0\\
# \end{cases}
# $$
# 
# where _remainder_$(x, y)$ refers to the remainder of _x_ divided by _y_.
# 
# As depicted by the definition, your implementation should be recursive.
# 
# You may assume that the provided numbers are valid.

# In[ ]:


def gcd(x, y): 

    if y == 0:
        return x
    else:
        return gcd(y, x % y)


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'GcdTest'], verbosity=2, exit=False)


# ## Exercise 4 (4 Points)
# Implement a class `Rectangle` that represents a rectangle object.
# 
# Your class should contain the following variables:
# * `a`, represents the height of the rectangle
# * `b`, represents the width of the rectangle
# 
# Your class should provide the following functions:
# * `__init__(self, a, b)`, the constructor taking two arguments $a$ and $b$ representing the height and width
# * `get_a(self)`, returns the height $a$
# * `get_b(self)`, returns the width $b$
# * `set_a(self, v)`, sets the height $a$ to the provided value $v$
# * `set_b(self, v)`, sets the width $b$ to the provided value $v$
# * `area(self)`, returns the area
# * `perimeter(self)`, returns the perimeter (Umfang)
# * `length_of_diagonal(self)`, returns the length of th diagonal

# In[ ]:


class Rectangle():
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def get_a(self):
        return self.a
    def get_b(self):
        return self.b
    def set_a(self,v):
        self.a = v
    def set_b(self,v):
        self.b = v
    def area(self):
        self.area = self.a * self.b
        return self.area
    def perimeter(self):
        self.perimeter = 2*(self.a +self.b)
        return self.perimeter
    def length_of_diagonal(self):
        return ( self.a**2 + self.b**2 )**0.5


# In[ ]:


# Run tests
unittest.main(argv=['ignored', '-v', 'RectangleTest'], verbosity=2, exit=False)

