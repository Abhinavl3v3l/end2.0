## BACKGROUND AND VERY BASICS

## What is a neural network neuron?

Neural network neuron is like small unit of neural network. 

It combines the weighted input and bias and gives an output. Finally an activation function is applied on the output to provide some non-linearity. Activation functions could be tanh, sigmoid, ReLU, etc.

$$
z=tanh(\sum_{i=1}^{n}w_{i}x_{i}+b)
$$



## What is the use of the learning rate?



Learning rate(lr) controls how much of the model weight to be adjusted wrt model gradients. Perhaps most difficult hyper parameter to tune.

- too small *lr* : small update to weights may result in getting stuck at local minima. Saddle Point
- too large lr : results in divergent behaviours and overshoots the global minima (PS there is no such thing as global minima, if we knew the global minima in advance, we don't even need to train the model)



## How are weights initialised?

Randomly

Weights in neural networks are optimised using gradient descent, say training. 

Wrong initial values can hinder the training process. Small init values can cause vanishing gradients, and too large values may lead to exploding gradients.

This problem can be solved, if

1. we keep mean of the init activations to zero, and
2. variance of these init activations should be same among all layers.

**Xavier initialization** deals with these problems nicely, considering number of layers and size of the layers.  Each weight is initialized with a small Gaussian value with mean = 0.0 and variance based on the fan-in and fan-out of the weight; fan-in is number of input nodes and fan_out is number of output/hidden nodes.

He initialization can be used with ReLU, while Xavier initialization works with tanh activations.

Weights are randomly initialized from a Gaussian distribution. Another method is glorot (or Xavier) initializer which samples from a distribution but truncates the values based on the kernel complexity. In glorot initialization, each weight is initialized with a small Gaussian value with mean = 0.0 and variance based on the fan-in and fan-out of the weight; fan-in is number of input nodes and fan_out is number of output/hidden nodes.



## What is "loss" in a neural network?

Loss is quantitative measure of how much the predictions differ from labels. Loss is basically prediction error and gradients are calculated for this loss function.

Loss value is calculated using a *loss function* or *cost function* which takes prediction and label values as input. Depending on the kind of problem we are trying to solve, different loss functions can be used. Mean Squared Error (MSE), Binary Crossentropy, Categorical Crossentropy are some common loss functions. 



1. Input

$$
x^i \in X | i = 1...m
$$

2. Output  
   $$
   y^i \in Y | i= 1...m
   $$

3. Where 
   $$
   X,Y \in \real
   $$

4. Training Set
   $$
   (x^i, y^i) | i =1...m
   $$

5. Given a Training set , learn a function 
   $$
   h: X \rightarrow Y
   $$
   so that h(x) is a good predictor of y

1. **Cost Function** -  

   1. A hypothesis function has variables : features and input data variables.

   2. Random features are usually assigned, data variables from data-set are constant.

   3. Function is modelled as the features are altered. Features are altered to increase accuracy using a feedback system. A feed back system can help direct the function to right direction.

   4. Accuracy of $h_\theta$/function is  measured using a **cost function**. 

      1. A cost function estimates by how far is the function from the expected output.
      2. Cost function : Its absolute difference between the predicted out $h_\theta(x)$ and output $(y)$.

   5. **Idea** - **A feedback system that measure how well our hypothesis/function fits into data**

   6. Cost Function or Squared error function or Mean Squared Function - 
      $$
      J(\theta) = \frac{1}{2m}\sum_{i=1}^m  (h_\theta(x)-y)^2
      $$

   7. In a gist this is how error and hence loss is calculated, and further rectified.





## What is the "chain rule" in gradient flow?

A neural network in whole is a composite function and to calculate the derivatives of a composite functions we need chain rule. 

Chain rules says that  if  a variable ***z\*** depends on the variable ***g***, which itself depends on the variable ***f\***, so that *z* and *g* are dependent variables, then *z*, via the intermediate variable of *g*, depends on *f* as well.
$$
f  \hookrightarrow g    \hookrightarrow z
$$


The chain rule is used for calculating the composite functions which are essentially nested equations.

if we have:
                                                                     <img src="https://latex.codecogs.com/gif.latex?x=f(t)" title="x=f(t)" />

<img src="https://latex.codecogs.com/gif.latex?y=g(t)" title="y=g(t)" />

<img src="https://latex.codecogs.com/gif.latex?z=h(x,y)" title="z=h(x,y)" />



Then by chain rule derivative of  z will be:



<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;z}{\partial&space;t}\;=\frac{\partial&space;z}{\partial&space;x}\frac{\partial&space;x}{\partial&space;t}&plus;\frac{\partial&space;z}{\partial&space;y}\frac{\partial&space;y}{\partial&space;t}" title="\frac{\partial z}{\partial t}\;=\frac{\partial z}{\partial x}\frac{\partial x}{\partial t}+\frac{\partial z}{\partial y}\frac{\partial y}{\partial t}" />

This is the core  and essential calculation of backward propagation.
