#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
from numpy import genfromtxt
from tabulate import tabulate
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# Setting up print settings, for nicely formatted output.  
print_settings = np.get_printoptions()
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
pd.set_option('display.width', 110)

# Variables being used for printing output 
i = 0
matrixColumns = []
while i < 100:
    matrixColumns.append(" ")
    i += 1


# # Homework 1 
# 
# ## Details 
# * Student: Segun Akinyemi 
# * Due Date: September 11th, 2019

# ## 1. Linear Algebra in Numpy

# 1. Create a random 100-by-100 matrix M, using numpy method "np.random.randn(100, 100)", where each element is drawn from a random normal distribution.

# In[2]:


M = np.random.randn(100, 100) 

# Data Frame library for printing Matrices nicely 
dataframe = pd.DataFrame(M, columns=matrixColumns, index=matrixColumns)
print("   100x100 Matrix, with each element drawn from a random normal distribution\n")
print(dataframe)


# 2. Calculate the mean and variance of all the elements in M 

# In[3]:


mean = np.mean(M)
variance = np.var(M)

print("    Mean and Variance of all elements in matrix M\n")
print("    Mean:", mean)
print("    Variance: ", variance)


# 3. Use "for loop" to calculate the mean and variance of each row of M.

# In[4]:


print("Mean and Variance for each row in the matrix.\n")

# Setting up a table for output
outputTable = PrettyTable()
outputTable.field_names = ["Row", "Mean", "Variance"]

# For loop calculating mean and variance for each row. 
index = 1
for row in M: 
    row_mean = np.mean(row)
    row_variance = np.var(row)
    outputTable.add_row([index, str(row_mean), str(row_variance)])
    index+=1

print(outputTable)


# 4. Use matrix operation instead of "for loop" to calculate the mean of each row of M, hint: create a vector of ones using np.ones(100, 1)?

# In[5]:


# Matrix Operation to calculate mean of each row
P = np.ones((100, 1))
PX = np.matmul(M, P)
row_means = PX/100

# Setting up a table to print output nicely. Not related to calculation
rowMeansTable = PrettyTable()
rowMeansTable.field_names = ["Row", "Mean"]
index = 1
for mean in row_means:
    rowMeansTable.add_row([index, mean])
    index += 1

print(rowMeansTable) 


# 5. Calculate the inverse matrix M<sup>-1</sup>

# In[6]:


M_inverse = linalg.inv(M)


# 5. Verify that M<sup>-1</sup>M = MM<sup>-1</sup> = I. Are the off-diagnoal elements exactly 0, why? 

# In[7]:


leftIdentity = np.dot(M_inverse, M)
rightIdentity = np.dot(M, M_inverse)

print("   Below is verification that the dot products match an identity matrix. np.allclose returns true if two")
print("   matrices are element wise equal. np.identity gives an identity matrix (ones diagnoal, zero elsewhere)")
print("\n")
print("  ", np.allclose(leftIdentity, np.identity(100)))
print("  ", np.allclose(rightIdentity, np.identity(100)))
print("\n")
print("   Verification above. The off diagnoal elements are not exactly zero. This is because a numeric calculation") 
print("   is by nature never perfect, so the values are extremely close to zero, but not zero exactly.")


# ## 2. Probability Distribution

# You have recently joined a data science team and working on a project that needs to simulate 5 types of distributions (Bernoulli, Poisson, Gaussian, uniform and Rolling-Dice distribution). Your teammate
# provides you with a simulated data sample `sample_trials.csv`. In the file, each column contains 5000 numbers drawn from one of the 5 distributions. However, the columns are not labeled properly and you have to figure out the labels yourself as your teammate is on vacation.

# In[8]:


trial_data = genfromtxt('sample_trials.csv', delimiter=',', skip_header=1)


# 1. Do the columns have discrete value or continuous value? How many unique values does each column have?

# In[9]:


unique_column1 = np.unique(trial_data[:, 0])
unique_column2 = np.unique(trial_data[:, 1])
unique_column3 = np.unique(trial_data[:, 2])
unique_column4 = np.unique(trial_data[:, 3])
unique_column5 = np.unique(trial_data[:, 4])

print("\n")
print("    Column 1 is Discrete and has", unique_column1.size, "unique values")
print("    Column 2 is Discrete and has", unique_column2.size, "unique values")
print("    Column 3 is Continuous and has", unique_column3.size, "unique values")
print("    Column 4 is Continuous and has", unique_column4.size, "unique values")
print("    Column 5 is Discrete and has", unique_column5.size, "unique values")


# 2. What are the min, max, mean, variance of the columns?

# In[10]:


# Calculations 
Mins = np.min(trial_data, axis=0)
Max = np.max(trial_data, axis=0)
Means = np.mean(trial_data, axis=0)
Variances = np.var(trial_data, axis=0)

# Combining all columns, to get the values for the entire data set. 
allColumns = np.column_stack((trial_data[:, 0], trial_data[:, 1], trial_data[:, 2], trial_data[:, 3], trial_data[:, 4]))

# Output table setup
calcsOutputTable = PrettyTable()
calcsOutputTable.field_names = ["Column", "Min", "Max", "Mean", "Variance"]
calcsOutputTable.add_row(["All", np.min(allColumns), np.max(allColumns), np.mean(allColumns), np.var(allColumns)])
calcsOutputTable.add_row([1, Mins[0], Max[0], Means[0], Variances[0]])
calcsOutputTable.add_row([2, Mins[1], Max[1], Means[1], Variances[1]])
calcsOutputTable.add_row([3, Mins[2], Max[2], Means[2], Variances[2]])
calcsOutputTable.add_row([4, Mins[3], Max[3], Means[3], Variances[3]])
calcsOutputTable.add_row([5, Mins[4], Max[4], Means[4], Variances[4]])

# Printing Output
print("\n")
print("Note: The first row shows the relevant calculation throughout all of the columns as a whole.")
print(calcsOutputTable)


# 3. Investigate the distribution of each column by plotting the histograms. Make sure you choose the bin size properly.

# In[11]:


# Column 1 Histogram
plt.hist(trial_data[:, 0], bins=range(int(Mins[0]), int(Max[0])), rwidth=0.9)
plt.title("Column 1 Histogram")
plt.show()

# Column 2 Histogram
plt.hist(trial_data[:, 1], bins=np.arange(1, 7, 0.9), rwidth=0.9)
plt.title("Column 2 Histogram")
plt.show()

# Column 3 Histogram
plt.hist(trial_data[:, 2], bins="auto", rwidth=0.9)
plt.title("Column 3 Histogram")
plt.show()

# Column 4 Histogram
plt.hist(trial_data[:, 3], bins="auto", rwidth=0.9)
plt.title("Column 4 Histogram")
plt.show()

# Column 5 Histogram
plt.hist(trial_data[:, 4], bins="auto", rwidth=0.9)
plt.title("Column 5 Histogram")
plt.show()


# 4.  Based on the analysis above, label each column with the distribution name and explain why. 

# * **Column 1:** Poisson Distribution
#     * **Explanation:** A Poisson distribution is discrete, which matches the discrete nature of Column 1 as discoverd in an earlier step. Furthermore, a Poisson random variable is always greater than or equal to zero, which matches the data observed in Column 1 that has a minimum value of 0 and a maximum value of 8. Furthermore, the Histogram for the plotted Column most closely resembles the shape of a Poisson distrbution, more so then any of the other columns
# 
# 
# * **Column 2:** Rolling Dice Distribution
#     * **Explanation:** Looking at the data for Column 2, we can see that it most closely matches the Rolling Dice distribution, more than any of the other columns. To begin, the minumm value is the integer 1, and the maximum value the integer 6. Those values are what you would expect given a set of data representing a 6 sided die, which would have little reason to record floating point values in it's data set. Column 2 is also discrete, having 6 unique values that repeat many times over within the data set, this is more evidence solidfying this as a Rolling Dice distribution. A dice has 6 sides, with 6 possible values, and a data set consisting of dice rolls would certainly have those values repeating many times over, but never going outside the interval of 1 - 6 inclusive. 
# 
# 
# * **Column 3:** Uniform Distribution
#     * **Explanation:** Although it's not a perfect fit, Column 3 most closley resembles a Uniform probability distribution. This is obvious first by observing it's histogram. In a Uniform distribution, the data follows a pattern where each group has approxtimately the same number of it's values occuring throughout the data set. This presents itself most often as a series of bars that are similar to one another in height, as can be seen in the plotted histogram of Column 3. Every bar in the histogram is around the same height. If we assume that the data set for Column 3 is real world data, then we can further identify it as a uniform distribution. In a real world data set, we're not going to get perfectly evenly distributed values across our interval, but we may very well get something that's close. Looking at the data for Column 3, we see that it is continous with some 5,000 unique values. The Uniform distribution is by nature continuous, which is additional support for this being a Uniform distribution. 
# 
# 
# * **Column 4:** Normal (or Gaussian) Distribution
#     * **Explanation:** Right off the bat, we know that the Normal distribution has a shape resembling that of a bell shaped curve. Of all of the columns, the plotted hisogram for Column 4 matches this shape most accurately. Another property of the Normal distribution is the apperance of a sort of center point in the data where much of the data clusters, with fewer and fewer values appearing as you depart from this center in either direction. We can see the center like point of Column 4 on it's plotted histogram, that being about 0 or close to it. As values depart from zero in either direction, they are less and less frequent, which matches the expectations of a Normally distrbuted data set. 
# 
# 
# * **Column 5:** Bernoulli Distribution
#     * **Explanation:** The shape of the plotted histogram of Column 5 most closely matches that of the Bernoulli distribution. This becomes more apparent when you start looking at the data for Column 5. A Bernoulli distribution has only two possible outcomes, success (most often times denoted as 1) or failure (most often times denoted as 0). As we discovered in earlier problems, Column 5 is discrete having only 2 unique values, which serves as more evidence that this is a Bernoulli distribution. We can see the min value of Column 5 is 0, and the max value is 1, with their being no instance of any values besides 0 and 1 in the entirety of the data set. This is exactly what you would expect with a true/false or success/failure real world data set based on the Bernoulli distribution. With this evidence, it is apparent that Column 5 belongs to the Bernoulli distribution. 

# 5. Knowing the mean, variance, write down the formulas of the distributions
# 
#   Column 1 (Poisson) Formula
#     * Poisson Formula: P(x; μ) = (e<sup>-μ</sup>) (μ<sup>x</sup>) / x!
#     * Applied To Column 1: P(5000; 1.9898) = (e<sup>-1.9898</sup>) (1.9898<sup>5000</sup>) / 5000!
# 
#   Column 2 (Rolling Dice) Formula
#     * No Formula? 
# 
#   Column 3 (Uniform) Formula
#     * Uniform Formula: `f(x) = 1 / (b - a)` where b is the max and a is the min.  
#     * Applied to Column 3: `f(x) = 1 / (0.9998915976210577 - 0.00037136174624674556)`
#     
#   Column 4 (Normal) Formula 
#     * Normal Formula: Y = { 1/[ σ * √(2π) ] } * e<sup>-(x - μ)2/2σ2</sup>
#     * Applied to Column 4: Y = { 1/[ 0.99 * √(2π) ] } * e<sup>-(x - 0.025)2/2(0.99)2</sup>
#     
#   Column 5 (Bernoulli) Formula
#     * Bernoulli Formula:  P(n) = p<sup>n</sup>(1-p)<sup>(1-n)</sup>. 
#     * Applied To Column 5: P(0)  = .695`<sup>0</sup>(1-.695)<sup>(1-0)</sup>
#     * Applied To Column 5: P(1) = .305`<sup>1</sup>(1-.305)<sup>(1-1)</sup>

# 6. Simulate another 5000 samples of Bernoulli distribution with the same set of parameters. Write it into a text file.

# In[12]:


bernoulli_samples = np.random.binomial(1, 0.305, 5000)
outfile = open("bernoulli-samples.txt", "w")
outfile.write(str(bernoulli_samples))

# Resetting default print options for the Kernel Unrelated to assignment. 
np.set_printoptions(print_settings)

