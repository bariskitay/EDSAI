#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from utils import f, analytical_grad_wrt_w1, analytical_grad_wrt_w2


# # Assigment 5

# ![Function Graph](function_graph.png)

# Given the function graph above.The gradient of the function 
# \begin{align}
# f(x, y; w_1, w_2) = \sin\left( \sqrt{
# 					\left(w_1x \right)^2 +
# 					\left(w_2y \right)^2 + 
# 					\left(w_1xw_2y \right)^2
# 							} \right)
# \end{align}
# can be computed by making use of the function graph and locally computing gradients at each node. 
# By doing this, we merely take advantage of the chain rule. Every node computes a forward pass and a backward pass.
# 
# The forward pass is just the function applied on the input. The backward pass to the input node is the partial derivative of the current node with respect to its inputs times the backward input it receives from earlier on. So, for example, the last two nodes will compute the following values:

# \begin{align}
# sin &: \begin{cases}
# 			\text{Forward Inputs:}\quad \quad sqrt.forward\_out \\
# 			\text{Backward Inputs:}\quad 1 \\
# 			\text{Forward Output:}\quad \sin(sqrt.forward\_out) \\
# 			\text{Derivative w.r.t. $sqrt$:}\quad \cos(sqrt.forward\_out)  \\
# 			\text{Backward pass to $sqrt$:}\quad \cos(sqrt.forward\_out) \times 1  \\
# 	       \end{cases}\\[2em]
# %
# sqrt &: \begin{cases}
# 			\text{Forward Inputs:}\quad \quad add_2.forward\_out \\
# 			\text{Backward Inputs:}\quad  sin.backward\_out\\
# 			\text{Forward Output:}\quad \sqrt{add_2.forward\_out} \\
# 			\text{Derivative w.r.t. $add_2$:}\quad 1/{(\sqrt{add_2.forward\_out})}  \\
# 			\text{Backward pass to $add_2$:}\quad 1/{(\sqrt{add_2.forward\_out})} \times \, \text{backward input}  \\
# 	       \end{cases}\\
# \end{align}
# 

# Here, the "." refers to accessing the nodes computed values, so $sqrt.forward\_out$ is just the output of the $sqrt$ node in the forward pass.

# ## Exercise 1

# Look at the following example for the addition node in the graph. Use the same structure to fill the gaps in the other nodes provided below. 
# 
# (Gaps are marked by TODO).
# 
# Note that it is not important that you understand the concept of a class in Python. Just think of it as a box which holds certain values, for example the local gradients with respect to the input nodes.
# 

# In[3]:


class AddNode:
    
    def __init__(self):
        # The __init__ function is called when you create a node, see below when we construct add1, for example.
        # All functions of our node will always refer to the node they represent by 'self'.
        # Hence, in the following, we just set the local gradients of the newly created node to None, as
        # the node did not process any input yet.
        
        self.local_gradient_input_1 = None
        self.local_gradient_input_2 = None

    def forward(self, input_node1, input_node2):
        # Compute the local gradients with respect to the input variables.
        # The function add(a, b) = a + b has a derivative of 1 with respect to a and b
        self.local_gradient_input_1 = 1
        self.local_gradient_input_2 = 1
        
        # If you call the forward function of the node, (see later with add1.forward(a, b)),
        # it will return a + b. This is the message that is sent to all subsequent nodes that 
        # get input from this node.
        return input_node1 + input_node2
    
    def backward(self, backward_input):
        # Make sure the local gradient is set. 
        # This means that we need to do a forward pass before a backward pass.
        assert self.local_gradient_input_1 is not None and self.local_gradient_input_2 is not None 
        
        # Now, we make use of the chain rule. Nodes later in the graph will provide the current value
        # of the gradient, which has to be updated. 
        
        # Hence, we multiply the local gradients with respect to the input nodes with the incoming backward signal
        output_to_node_1 = backward_input * self.local_gradient_input_1
        output_to_node_2 = backward_input * self.local_gradient_input_2
        
        # Each input to our node will get a backward signal, that depends on the gradient with 
        # respect to the input. In the case of AddNode, these are the same for both inputs.
        # Of course, this is not generally the case for more complicated functions.
        return output_to_node_1, output_to_node_2
    
    


# ### a)

# Implement the forward and backward function for the multiplication node.

# In[5]:


class MulNode:
    
    def __init__(self):
        # The __init__ function is called when you create a node, see below when we construct add1, for example.
        # All functions of our node will always refer to the node they represent by 'self'.
        # Hence, in the following, we just set the local gradients of the newly created node to None, as
        # the node did not process any input yet.
        self.local_gradient_input_1 = None
        self.local_gradient_input_2 = None

    def forward(self, input_node1, input_node2):
        # Compute the local gradients with respect to the input variables.
        # -> What is the derivative of the function with respect to its inputs?
        self.local_gradient_input_1 = input_node2
        self.local_gradient_input_2 = input_node1
        
        return input_node1 * input_node2
    
    def backward(self, backward_input):
        # Make sure the local gradient is set. 
        # This means that we need to do a forward pass before a backward pass.
        assert self.local_gradient_input_1 is not None and self.local_gradient_input_2 is not None 
        
        # Multiply the local gradients with respect to the input nodes with the incoming backward signal
        output_to_node_1 = backward_input * self.local_gradient_input_1
        output_to_node_2 = backward_input * self.local_gradient_input_2
        
        return output_to_node_1 , output_to_node_2
    
    


# In[7]:


class SquareNode:
    
    def __init__(self):
        # The __init__ function is called when you create a node, see below when we construct add1, for example.
        # All functions of our node will always refer to the node they represent by 'self'.
        # Hence, in the following, we just set the local gradients of the newly created node to None, as
        # the node did not process any input yet.
        self.local_gradient_input_1 = None
        self.local_gradient_input_2 = None

    def forward(self, input_node1):
        # Compute the local gradients with respect to the input variables.
        # -> What is the derivative of the function with respect to its inputs?
        self.local_gradient_input_1 = 2 * input_node1
        
        return input_node1 ** 2 
    
    def backward(self, backward_input):
        # Make sure the local gradient is set. 
        # This means that we need to do a forward pass before a backward pass.
        assert self.local_gradient_input_1 is not None
        
        # Multiply the local gradients with respect to the input nodes with the incoming backward signal
        output_to_node_1 = self.local_gradient_input_1 * backward_input
        
        return output_to_node_1
    
    


# In[9]:


class SquareRootNode:
    
    def __init__(self):
        # The __init__ function is called when you create a node, see below when we construct add1, for example.
        # All functions of our node will always refer to the node they represent by 'self'.
        # Hence, in the following, we just set the local gradients of the newly created node to None, as
        # the node did not process any input yet.
        self.local_gradient_input_1 = None
        self.local_gradient_input_2 = None

    def forward(self, input_node1):
        # Compute the local gradients with respect to the input variables.
        # -> What is the derivative of the function with respect to its inputs?
        # For efficiency, we can first compute the output and use it to set the gradient. 
        # Please write the gradient as a function of the output.
        out = math.sqrt(input_node1)
        self.local_gradient_input_1 = (0.5 / np.sqrt(input_node1))
        return out
    
    def backward(self, backward_input):
        # Make sure the local gradient is set. 
        # This means that we need to do a forward pass before a backward pass.
        assert self.local_gradient_input_1 is not None
        # Multiply the local gradients with respect to the input nodes with the incoming backward signal
        output_to_node_1 = backward_input * self.local_gradient_input_1
        
        return output_to_node_1
    
    


# In[11]:


class SinusNode:
    
    def __init__(self):
        # The __init__ function is called when you create a node, see below when we construct add1, for example.
        # All functions of our node will always refer to the node they represent by 'self'.
        # Hence, in the following, we just set the local gradients of the newly created node to None, as
        # the node did not process any input yet.
        self.local_gradient_input_1 = None
        self.local_gradient_input_2 = None

    def forward(self, input_node1):
        # Compute the local gradients with respect to the input variables.
        # -> What is the derivative of the function with respect to its inputs?
        # Hint: For this function, (not for the previous ones!), 
        # you can use the functions provided in the math package. 
        # You can use the functions by calling math.sin, math.cos, math.sqrt etc...
        self.local_gradient_input_1 = np.cos(input_node1)
        
        return np.sin(input_node1)
    
    def backward(self, backward_input):
        # Make sure the local gradient is set. 
        # This means that we need to do a forward pass before a backward pass.
        assert self.local_gradient_input_1 is not None
        # Multiply the local gradients with respect to the input nodes with the incoming backward signal
        output_to_node_1 = backward_input * self.local_gradient_input_1
        
        return output_to_node_1
    
    


# #### Creating the nodes of the graph

# In[13]:


mul1 = MulNode()
mul2 = MulNode()
mul3 = MulNode()
sq1 = SquareNode()
sq2 = SquareNode()
sq3 = SquareNode()
add1 = AddNode()
add2 = AddNode()
sqrt = SquareRootNode()
sin = SinusNode()


# Hint: You can verify your implementation by comparing the forward and backward outputs with your analytical solutions.
# Look at the following example:

# ##### Forward pass

# In[14]:


add1.forward(5, 6) == 5 + 6


# ##### Backward pass

# Make sure you first do a forward pass!

# In[15]:


# If the gradient of the preceding node is 1, the gradient with respect to either input will be 1, too.
# Note that the node returns 2 values, one for each node. We can split them up as follows
gradient_wrt_node1, gradient_wrt_node2 = add1.backward(1)
print(gradient_wrt_node1 == 1)
print(gradient_wrt_node2 == 1)
# More generally, as is easy to see, the addnode will just forward the gradients in the backward pass
# to both input nodes.
gradient_wrt_node1, gradient_wrt_node2 = add1.backward(5)
print(gradient_wrt_node1 == 5)
print(gradient_wrt_node2 == 5)


# ### b) Creating the graph

# ### Forward pass

# In[16]:


def graph_forward(x, y, w1, w2):
    # Complete the forward pass according to the graph structure shown above.
    mul1_out = mul1.forward(w1, x)
    mul2_out = mul2.forward(w2, y)
    mul3_out = mul3.forward(mul1_out, mul2_out)
    sq1_out = sq1.forward(mul1_out)
    sq2_out = sq2.forward(mul3_out)
    sq3_out = sq3.forward(mul2_out)
    add1_out = add1.forward(sq1_out, sq2_out)
    add2_out = add2.forward(add1_out, sq3_out)
    sqrt_out = sqrt.forward(add2_out)
    sin_out = sin.forward(sqrt_out)
    return sin_out


# To test your implementation of the forward pass of the graph, we provided the analytical implementation. 

# <font color="red"><b>ATTENTION!</b></font> 
# 
# If you change the implementation of any of the nodes, you need to create the instantiations of the classes again (i.e., you need to run the cell with mul1 = ... again.) Otherwise, the graph will not include the update that you made to the Nodes.

# In[17]:


# Creating 4 positive random numbers between 1 and 2. (np.random.random returns a number between 0 and 1)
x, y, w1, w2 = np.random.random(4) + 1
print(x, y, w1, w2)


# In[18]:


# f(x, y, w1, w2) is the direct computation of the result. Make sure that graph forward matches this!
print(graph_forward(x, y, w1, w2), f(x, y, w1, w2))


# (The following picture is the same as above. Just so you do not need to scroll if you want to see the graph.)

# ![Function Graph](function_graph.png)

# ### Backward pass

# Hint: The output of some nodes serves as input to more than 1 subsequent node. How do we combine the gradients coming from the different nodes? Look at this example:
# $$
# f(g(x)) = s(g(x)) + t(g(x))\\
# \\
# \Rightarrow \frac{\partial f(g(x))}{\partial g(x)} = \quad?
# $$

# In[19]:


def graph_backward():
    # Complete the backward pass according to the graph structure shown above. 
    init_grad = 1
    grad_wrt_sqrt = sin.backward(init_grad)
    grad_wrt_add2 = sqrt.backward(grad_wrt_sqrt)
    grad_wrt_add1, grad_wrt_sqr = add2.backward(grad_wrt_add2)
    grad_wrt_mul2_1 = sq3.backward(grad_wrt_sqr)
    grad_wrt_sq1, grad_wrt_sq2 = add1.backward(grad_wrt_add1)
    grad_wrt_mul3 = sq2.backward(grad_wrt_sq2)
    grad_wrt_mul1_1, grad_wrt_mul2_2 = mul3.backward(grad_wrt_mul3)
    grad_wrt_mul1_2 = sq1.backward(grad_wrt_sq1)

    grad_wrt_mul1 = grad_wrt_mul1_1 + grad_wrt_mul1_2
    grad_wrt_mul2 = grad_wrt_mul2_1 + grad_wrt_mul2_2
    grad_wrt_w1, grad_wrt_x = mul1.backward(grad_wrt_mul1)
    grad_wrt_w2, grad_wrt_y = mul2.backward(grad_wrt_mul2)
    return grad_wrt_w1, grad_wrt_w2


# To test your implementation of the backward pass, we provided the analytical implementation. 

# In[21]:


grad_wrt_w1, grad_wrt_w2 = graph_backward()


# Note that due to numerical instabilities, the values might not be exactly the same.

# In[22]:


print((grad_wrt_w1, analytical_grad_wrt_w1(x, y, w1, w2)))


# In[23]:


print(np.allclose(grad_wrt_w1, analytical_grad_wrt_w1(x, y, w1, w2)))
print(np.allclose(grad_wrt_w2, analytical_grad_wrt_w2(x, y, w1, w2)))

