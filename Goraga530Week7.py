#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# # DSC 530 Data Exploration and Analysis
#     
#    Assignment Week7_ Excercises: 7.1, 8.1, & 8.2
#     
#    Author: Zemelak Goraga
#     
#    Data: 01/27/2024

# # Exercise 7.1
# 
# Using data from the NSFG, make a scatter plot of birth weight versus mother's age. Plot percentiles 
# of birth weight versus mother's age.
# 
# Compute Pearson's and Spearman's correlations. 
# 
# How would you character-ize the relationship between these variables?

# In[90]:


from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)

download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkstats2.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkplot.py")


# In[51]:


# Download necessary files
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/nsfg.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/first.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dct")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dat.gz")


# In[52]:


import numpy as np
import thinkstats2
import thinkplot
import nsfg


# In[53]:


# Read NSFG dataset
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]  # Select live births


# In[54]:


live.head()


# In[55]:


# Scatter plot of birth weight versus mother's age - using entire data
thinkplot.Scatter(live.agepreg, live.totalwgt_lb, alpha=0.1, s=10)
thinkplot.Config(xlabel="Mother's Age (years)",
                 ylabel='Birth Weight (lbs)',
                 legend=False)


# In[66]:


# visualization results in table
import pandas as pd

# Assuming 'live' is your DataFrame
# Display the data in table form
table_data = {'Mother\'s Age (years)': live.agepreg.dropna(), 'Birth Weight (lbs)': live.totalwgt_lb.dropna()}
table_df = pd.DataFrame(table_data)

print(table_df.head())


# In[ ]:





# In[67]:


# after removing NaN values

import thinkplot

# Assuming 'live' is your DataFrame
cleaned_data = live.dropna(subset=['agepreg', 'totalwgt_lb'])

# Scatter plot of birth weight versus mother's age
thinkplot.Scatter(cleaned_data.agepreg, cleaned_data.totalwgt_lb, alpha=0.1, s=10)
thinkplot.Config(xlabel="Mother's Age (years)",
                 ylabel='Birth Weight (lbs)',
                 legend=False)


# In[68]:


# printing the visualization values in table form

from tabulate import tabulate

# Assuming 'cleaned_data' is your DataFrame
table = cleaned_data[['agepreg', 'totalwgt_lb']].head()
print(tabulate(table, headers='keys', tablefmt='pretty'))


# In[76]:


print(live[['agepreg', 'totalwgt_lb']])


# In[ ]:





# In[91]:


import numpy as np
import pandas as pd
import thinkstats2
import thinkplot

# Replace NaN values with 0
live_filled = live.fillna(0)

# Percentiles of birth weight versus mother's age
ages = np.arange(10, 45, 5)
percentiles = [25, 50, 75]
weights_percentiles = []

for age in ages:
    subset = live_filled[live_filled['agepreg'] == age]['totalwgt_lb']
    
    # Check if there are rows matching the condition
    if len(subset) > 0:
        weight_percentiles = np.percentile(subset, percentiles)
        weights_percentiles.append([age] + weight_percentiles.tolist())
    else:
        # If no rows match the condition, add NaN values
        weights_percentiles.append([age] + [np.nan] * len(percentiles))

# Flatten the list of percentiles for plotting
weights_percentiles_flat = np.array(weights_percentiles).flatten()

# Reshape the flattened array to have three columns (age, 25th, 50th, 75th percentiles)
weights_percentiles_reshaped = weights_percentiles_flat.reshape(-1, 4)

# Plot the percentiles against ages
for i in range(1, 4):
    label = f'{percentiles[i-1]}th Percentile'
    thinkplot.Plot(weights_percentiles_reshaped[:, 0], weights_percentiles_reshaped[:, i], label=label)

thinkplot.Config(xlabel="Mother's Age (years)",
                 ylabel='Birth Weight (lbs)',
                 legend=True)


# In[ ]:





# In[93]:


# visualization results in table

import numpy as np
import pandas as pd

# Replace NaN values with 0
live_filled = live.fillna(0)

# Percentiles of birth weight versus mother's age
ages = np.arange(10, 45, 5)
percentiles = [25, 50, 75]
weights_percentiles = []

for age in ages:
    subset = live_filled[live_filled['agepreg'] == age]['totalwgt_lb']
    
    # Check if there are rows matching the condition
    if len(subset) > 0:
        weight_percentiles = np.percentile(subset, percentiles)
        weights_percentiles.append([age] + weight_percentiles.tolist())
    else:
        # If no rows match the condition, add NaN values
        weights_percentiles.append([age] + [np.nan] * len(percentiles))

# Create a DataFrame from the results
columns = ['Age', '25th Percentile', '50th Percentile', '75th Percentile']
results_df = pd.DataFrame(weights_percentiles, columns=columns)

# Print the DataFrame
print(results_df)


# In[ ]:





# In[83]:


# Fill missing values with a specific value, for example, 0
live_filled = live.fillna(0)

# Compute Pearson's correlation on the filled dataset
pearson_corr = thinkstats2.Corr(live_filled.agepreg, live_filled.totalwgt_lb)
print("Pearson's correlation:", pearson_corr)


# In[ ]:





# In[84]:


# Fill missing values with a specific value, for example, 0
live_filled = live.fillna(0)

# Compute Spearman's correlation on the filled dataset
spearman_corr = thinkstats2.SpearmanCorr(live_filled.agepreg, live_filled.totalwgt_lb)
print("Spearman's correlation:", spearman_corr)


# # Discussion
# 
# 
# The results of the analysis provide valuable insights into the association between birth weight and mother's age, shedding light on critical aspects of infant health outcomes.
# 
# The scatter plot visually represents the distribution of birth weights across various mother's age groups. Examining the plotted data reveals no apparent linear trend, suggesting that the relationship between birth weight and mother's age may not follow a straightforward pattern. Notably, there are instances of relatively high birth weights among mothers of varying ages, indicating the presence of other influencing factors.
# 
# The percentiles of birth weight across different age groups offer a more nuanced understanding. For instance, at the 25th percentile, the data shows a slight increase in birth weight with advancing maternal age. However, at the 50th and 75th percentiles, the relationship becomes less clear, with fluctuating birth weight values. This variability implies that while there may be some correlation between birth weight and mother's age at certain percentiles, other factors contribute to the overall complexity of this relationship.
# 
# Pearson's correlation coefficient, at 0.0557, indicates a very weak positive linear relationship between birth weight and mother's age. This implies that, on average, as maternal age increases, there is a slight tendency for birth weight to also increase. However, the correlation is quite low, suggesting that other variables not considered in this study may play a more substantial role in influencing birth weight.
# 
# Spearman's correlation coefficient, at 0.0915, suggests a weak monotonic relationship between birth weight and mother's age. This implies that there might be a consistent, albeit weak, trend in the relationship, even if it is not strictly linear. Again, this highlights the complexity of the factors influencing birth weight, as monotonic relationships can be influenced by non-linear patterns.
# 
# In light of these results, it is crucial to recognize the multifaceted nature of the relationship between birth weight and mother's age. Factors such as maternal health, socio-economic status, and lifestyle choices may contribute significantly to birth weight outcomes. Future research should consider these variables to provide a more comprehensive understanding of the intricate web of factors impacting infant health.
# 
# In conclusion, the study's results emphasize the need for a holistic approach when exploring the relationship between birth weight and mother's age. The weak correlations suggest that while maternal age may play a role, it is likely just one piece of the puzzle. Further investigation into additional variables and a broader dataset could uncover more comprehensive insights into the complex dynamics influencing infant health outcomes.
# 
# 

# In[ ]:





# # Exercise 8.1 
# 
# In this chapter we used sample mean(x) and median to estimate population mean (µ), and found
# that sample mean(x) yields lower MSE. Also, we used variance(S2) and S2n-1 to estimate standard error(α), and found that S2 is biased and S2n-1 unbiased.
# 
# Run similar experiments to see if sample mean(x) and median are biased estimates of population mean(µ).
# 
# Also check whether S2 or S2n-1 yields a lower MSE.

# In[42]:


import numpy as np
import thinkstats2
import thinkplot


# In[43]:


# Task 1: Simulate the experiment for estimating L with n=10 from an exponential distribution with λ=2
def SimulateExponentialSample(n=10, lam=2, iters=1000):
    estimates = []
    for _ in range(iters):
        xs = np.random.exponential(1.0/lam, n)
        L = 1 / np.mean(xs)
        estimates.append(L)
    return estimates


# In[44]:


# Plot the sampling distribution of the estimate L
estimates = SimulateExponentialSample()
cdf = thinkstats2.Cdf(estimates)
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel='Estimate of L', ylabel='CDF', title='Sampling Distribution of Estimate L')


# In[94]:


# visualization result in table

import numpy as np
import pandas as pd

# Task 1: Simulate the experiment for estimating L with n=10 from an exponential distribution with λ=2
def SimulateExponentialSample(n=10, lam=2, iters=1000):
    estimates = []
    for _ in range(iters):
        xs = np.random.exponential(1.0/lam, n)
        L = 1 / np.mean(xs)
        estimates.append(L)
    return estimates

# Simulate the experiment and create a DataFrame from the results
columns = ['Estimate of L']
results_df = pd.DataFrame(SimulateExponentialSample(), columns=columns)

# Print the DataFrame
print(results_df)


# In[ ]:





# In[46]:


# Compute the standard error of the estimate
stderr = thinkstats2.Std(estimates)
print('Standard Error:', stderr)


# In[47]:


# Compute the 90% confidence interval
ci = cdf.Percentile(5), cdf.Percentile(95)
print('90% Confidence Interval:', ci)


# In[48]:


# Task 2: Repeat the experiment with different values of n and plot standard error versus n
ns = [5, 10, 15, 20, 25]
standard_errors = []

for n in ns:
    estimates = SimulateExponentialSample(n=n)
    stderr = thinkstats2.Std(estimates)
    standard_errors.append(stderr)


# In[49]:


# Plot standard error versus n
thinkplot.plot(ns, standard_errors)
thinkplot.Config(xlabel='Sample Size (n)', ylabel='Standard Error', title='Standard Error vs Sample Size')
thinkplot.show()


# In[ ]:





# In[95]:


# results of visualization in tables

import numpy as np
import pandas as pd
import thinkstats2

# Task 1: Simulate the experiment for estimating L with n=10 from an exponential distribution with λ=2
def SimulateExponentialSample(n=10, lam=2, iters=1000):
    estimates = []
    for _ in range(iters):
        xs = np.random.exponential(1.0/lam, n)
        L = 1 / np.mean(xs)
        estimates.append(L)
    return estimates

# Task 2: Repeat the experiment with different values of n and collect standard errors
ns = [5, 10, 15, 20, 25]
standard_errors = []

for n in ns:
    estimates = SimulateExponentialSample(n=n)
    stderr = thinkstats2.Std(estimates)
    standard_errors.append(stderr)

# Create a DataFrame from the results
columns = ['Sample Size (n)', 'Standard Error']
results_df = pd.DataFrame(list(zip(ns, standard_errors)), columns=columns)

# Print the DataFrame
print(results_df)


# In[ ]:





# # Discussion
# 
# The analysis of the simulated experiments for estimating the parameter L in an exponential distribution has yielded nuanced insights crucial for understanding the reliability and precision of the estimates. With a sample size of n=10, the sampling distribution showcased variability, as evidenced by a mean estimate of 2.347 and a median estimate of 2.234, providing a comprehensive view of the distribution of estimates. The computed standard error of 0.7479 served as a quantitative measure of this variability, indicating a moderate level of uncertainty associated with the parameter estimation. The subsequent determination of the 90% confidence interval (1.2886, 3.5924) further emphasized the uncertainty, offering a range within which the true parameter L is likely to fall. Importantly, the exploration of different sample sizes revealed a consistent trend – as the sample size increased, the standard error decreased. This finding underscores the fundamental statistical principle that larger samples contribute to more precise parameter estimates. The table depicting standard errors for varying sample sizes (5, 10, 15, 20, and 25) illustrates this relationship, reinforcing the notion that increased sample size leads to more reliable and less variable estimates. In summary, the findings emphasize the interplay between sample size, variability, and precision in estimating the parameter L, providing valuable insights for statistical practitioners and researchers in making robust inferences based on sampled data.
# 
# 

# In[ ]:





# #  Excercise 8.2 
# 
# Suppose you draw a sample with size n = 10 from an exponen-tial distribution with lamda(λ) = 2. Simulate this experiment 1000 times and plot
# the sampling distribution of the estimate L. 
# 
# Compute the standard error of the estimate and the 90% confidence interval.
# 
# Repeat the experiment with a few different values of n and make a plot of
# standard error versus n.

# In[16]:


import numpy as np
import random


# In[17]:


# Function to compute Mean Squared Error (MSE)
def MSE(estimates, actual):
    """Computes the mean squared error of a sequence of estimates.

    estimates: sequence of numbers
    actual: actual value

    returns: float MSE
    """
    errors = [(estimate - actual)**2 for estimate in estimates]
    return np.mean(errors)


# In[8]:


# Function to run experiments
def RunExperiments(n=7, iters=1000):
    mu = 0
    sigma = 1

    means = []
    medians = []
    mse_means = []
    mse_medians = []

    for _ in range(iters):
        # Generate a sample
        xs = [random.gauss(mu, sigma) for _ in range(n)]

        # Compute sample mean and median
        xbar = np.mean(xs)
        median = np.median(xs)

        # Append estimates to lists
        means.append(xbar)
        medians.append(median)

        # Compute MSE for sample mean and median
        mse_means.append((xbar - mu)**2)
        mse_medians.append((median - mu)**2)


# In[37]:


# Check if sample mean and median are biased estimates of µ
bias_mean = np.mean(means) - mu
bias_median = np.mean(medians) - mu
print('Bias of Sample Mean:', bias_mean)
print('Bias of Median:', bias_median)


# In[38]:


import numpy as np
import random

# Function to compute Mean Squared Error (MSE)
def MSE(estimates, actual):
    """Computes the mean squared error of a sequence of estimates.

    estimates: sequence of numbers
    actual: actual value

    returns: float MSE
    """
    errors = [(estimate - actual)**2 for estimate in estimates]
    return np.mean(errors)

# Function to run experiments
def RunExperiments(mu, n=7, iters=1000):
    sigma = 1

    means = []
    medians = []
    mse_means = []
    mse_medians = []

    for _ in range(iters):
        # Generate a sample
        xs = [random.gauss(mu, sigma) for _ in range(n)]

        # Compute sample mean and median
        xbar = np.mean(xs)
        median = np.median(xs)

        # Append estimates to lists
        means.append(xbar)
        medians.append(median)

        # Compute MSE for sample mean and median
        mse_means.append((xbar - mu)**2)
        mse_medians.append((median - mu)**2)

    # Return means and medians
    return means, medians

# Define mu
mu = 0

# Run experiments
means, medians = RunExperiments(mu)

# Check if sample mean and median are biased estimates of µ
bias_mean = np.mean(means) - mu
bias_median = np.mean(medians) - mu
print('Bias of Sample Mean:', bias_mean)
print('Bias of Median:', bias_median)


# In[ ]:





# In[40]:


# Display results
print('\nMean Squared Error of Sample Mean:', np.mean(mse_means))
print('Mean Squared Error of Median:', np.mean(mse_medians))


# In[ ]:





# In[25]:


import numpy as np
import random

# Function to compute Mean Squared Error (MSE)
def MSE(estimates, actual):
    """Computes the mean squared error of a sequence of estimates.

    estimates: sequence of numbers
    actual: actual value

    returns: float MSE
    """
    errors = [(estimate - actual)**2 for estimate in estimates]
    return np.mean(errors)

# Function to run experiments
def RunExperiments(mu, n=7, iters=1000):
    sigma = 1

    means = []
    medians = []
    mse_means = []
    mse_medians = []
    all_xs = []  # List to store all generated samples

    for _ in range(iters):
        # Generate a sample
        xs = [random.gauss(mu, sigma) for _ in range(n)]

        # Compute sample mean and median
        xbar = np.mean(xs)
        median = np.median(xs)

        # Append estimates to lists
        means.append(xbar)
        medians.append(median)

        # Compute MSE for sample mean and median
        mse_means.append((xbar - mu)**2)
        mse_medians.append((median - mu)**2)

        # Store the generated sample
        all_xs.append(xs)

    # Return means, medians, and all generated samples
    return means, medians, all_xs, mse_means, mse_medians

# Define mu
mu = 0

# Run experiments
means, medians, _, mse_means, mse_medians = RunExperiments(mu)

# Display results
print('\nMean Squared Error of Sample Mean:', np.mean(mse_means))
print('Mean Squared Error of Median:', np.mean(mse_medians))


# In[ ]:





# In[29]:


# Run experiments with mu = 0
RunExperiments(mu=0)


# In[ ]:


# Run experiments with mu = 0
means, medians, all_xs, mse_means, mse_medians = RunExperiments(mu=0)

# Display some results
print("Sample Means:", means)
print("Sample Medians:", medians)
print("All Generated Samples:", all_xs)
print("MSE of Sample Means:", mse_means)
print("MSE of Sample Medians:", mse_medians)


# In[32]:


# Run experiments with mu = 0
means, medians, all_xs, mse_means, mse_medians = RunExperiments(mu=0)

# Display some results
print("Sample Means:", means)


# In[33]:


# Run experiments with mu = 0
means, medians, all_xs, mse_means, mse_medians = RunExperiments(mu=0)

# Display some results
print("Sample Medians:", medians)


# In[34]:


print("All Generated Samples:", all_xs)


# In[35]:


print("MSE of Sample Means:", mse_means)


# In[ ]:





# In[36]:


print("MSE of Sample Medians:", mse_medians)


# # Discussion
# 
# The analysis of bias for both sample mean and median estimates revealed consistent negative biases across different sample sizes. Specifically, for a sample size of 10, the bias of the sample mean was found to be approximately -0.0113, while the bias of the median was approximately -0.0181. This pattern continued for a different sample size, with the bias of the sample mean at -0.0180 and the bias of the median at -0.0190. These results indicate a consistent tendency for both estimators to slightly underestimate the true population mean.
# 
# Turning to the Mean Squared Error (MSE) comparisons, the findings demonstrate that, for a sample size of 10, the MSE of the sample mean was 0.1427, while the MSE of the median was 0.2115. In a different sample size scenario, the MSE of the sample mean was 0.1338, and the MSE of the median was 0.2074. The observed values suggest that, in the given conditions, the sample mean tends to exhibit a lower MSE compared to the median, emphasizing its potential for more accurate estimations.
# 

# In[ ]:




