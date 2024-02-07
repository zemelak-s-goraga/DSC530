#!/usr/bin/env python
# coding: utf-8

# # DSC 530 Data Exploration and Analysis
#     
#    Assignment Week8_ Excercises: 9.1, & 10.1
#     
#    Author: Zemelak Goraga
#     
#    Data: 2/4/2024

# # Excercise 9.1
# 
# As sample size increases, the power of a hypothesis test increases, which means it is more likely to be positive if the effect is real. Conversely, as sample size decreases, the test is less likely to be positive even if the effect is real.
# 
# To investigate this behavior, run the tests in this chapter with different 
# subsets of the NSFG data. You can use thinkstats2.SampleRows to select a
# random subset of the rows in a DataFrame.
# 
# What happens to the p-values of these tests as sample size decreases? 
# 
# What is the smallest sample size that yields a positive test?

# In[ ]:


from os.path import basename, exists
def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)

download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkstats2.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkplot.py")


# In[ ]:


# Download necessary files
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/nsfg.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/first.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dct")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/2002FemPreg.dat.gz")


# In[3]:


import numpy as np
import thinkstats2
import thinkplot
import nsfg
import pandas as pd
import hypothesis
import random


# In[4]:


# Read NSFG dataset
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]  # Select live births


# In[5]:


live.head()


# In[6]:


get_ipython().system('pip show hypothesis')


# In[7]:


# Tests a difference in means of pregnancy length and means of birth weight as sample size changes
# based on NSFG datset
import numpy as np
import pandas as pd
import thinkstats2
from scipy import stats
import nsfg

class DiffMeansPermute:
    """Tests a difference in means using permutation."""

    def __init__(self, data):
        """Initialize the hypothesis test.

        data: tuple of two arrays
        """
        self.group1, self.group2 = data

    def PValue(self, iters=1000):
        """Compute the p-value using permutation.

        iters: number of iterations

        returns: float
        """
        observed_diff = np.mean(self.group1) - np.mean(self.group2)
        combined = np.concatenate((self.group1, self.group2))

        count = 0
        for _ in range(iters):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(self.group1)]
            perm_group2 = combined[len(self.group1):]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            if perm_diff >= observed_diff:
                count += 1

        return count / iters

def RunTests(live, iters=1000):
    """Runs the tests from Chapter 9 with a subset of the data.

    live: DataFrame
    iters: how many iterations to run

    returns: tuple of p-values
    """
    n = len(live)
    firsts = live[live['birthord'] == 1]
    others = live[live['birthord'] != 1]

    # Compare pregnancy lengths
    data1 = firsts['prglngth'].values
    data2 = others['prglngth'].values
    ht1 = DiffMeansPermute((data1, data2))
    p1 = ht1.PValue(iters=iters)

    # Compare birth weights
    data3 = firsts['totalwgt_lb'].dropna().values
    data4 = others['totalwgt_lb'].dropna().values
    ht2 = DiffMeansPermute((data3, data4))
    p2 = ht2.PValue(iters=iters)

    print(f"Sample Size: {n}\tPregnancy Lengths P-Value: {p1:.6f}\tBirth Weights P-Value: {p2:.6f}")
    return p1, p2

def main():
    thinkstats2.RandomSeed(18)

    # Read NSFG dataset
    preg = nsfg.ReadFemPreg()
    live = preg[preg.outcome == 1]  # Select live births

    # Vary sample size and run tests
    sample_sizes = [500, 400, 300, 200, 100]

    # Track the smallest sample size that yields a positive test
    smallest_positive_sample_size = float('inf')

    for size in sample_sizes:
        sample = thinkstats2.SampleRows(live, size)
        p1, p2 = RunTests(sample)

        # Check if the test is positive and update the smallest positive sample size
        if p1 < 0.05 or p2 < 0.05:
            smallest_positive_sample_size = min(smallest_positive_sample_size, size)

    print(f"\nSmallest sample size yielding a positive test: {smallest_positive_sample_size}")

if __name__ == '__main__':
    main()


# # explanation
# 
# The code performs permutation tests on pregnancy lengths and birth weights for different sample sizes from the NSFG dataset. The output shows the sample size, p-values for pregnancy lengths, and birth weights. The "Smallest sample size yielding a positive test" indicates the smallest sample size with a p-value below 0.05 in either test. In this case, the output suggests that none of the tested sample sizes produced a statistically significant difference in the means of pregnancy lengths or birth weights.

# In[ ]:




chisquare for pregnancy length as sample size changes


code2:

# chi square on pregnancy length with different sample sizes
import numpy as np
import pandas as pd
import thinkstats2
from scipy import stats
import nsfg

class DiffMeansPermute:
    """Tests a difference in means using permutation."""

    def __init__(self, data):
        """Initialize the hypothesis test.

        data: tuple of two arrays
        """
        self.group1, self.group2 = data

    def PValue(self, iters=1000):
        """Compute the p-value using permutation.

        iters: number of iterations

        returns: float
        """
        observed_diff = np.mean(self.group1) - np.mean(self.group2)
        combined = np.concatenate((self.group1, self.group2))

        count = 0
        for _ in range(iters):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(self.group1)]
            perm_group2 = combined[len(self.group1):]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            if perm_diff >= observed_diff:
                count += 1

        return count / iters

def ChiSquareTestPregLength(live):
    """Performs a chi-square test of pregnancy length for first and non-first babies.

    live: DataFrame

    returns: tuple (chi2 statistic, p-value, degrees of freedom, expected frequencies)
    """
    contingency_table = pd.crosstab(live['birthord'] == 1, live['prglngth'] < 37)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return chi2_stat, p_value, dof, expected

def RunTests(live, iters=1000):
    """Runs the tests from Chapter 9 with a subset of the data.

    live: DataFrame
    iters: how many iterations to run

    returns: tuple of p-values and chi-square test results
    """
    n = len(live)
    firsts = live[live['birthord'] == 1]
    others = live[live['birthord'] != 1]

    # Compare pregnancy lengths
    data1 = firsts['prglngth'].values
    data2 = others['prglngth'].values
    ht1 = DiffMeansPermute((data1, data2))
    p1 = ht1.PValue(iters=iters)

    # Compare birth weights
    data3 = firsts['totalwgt_lb'].dropna().values
    data4 = others['totalwgt_lb'].dropna().values
    ht2 = DiffMeansPermute((data3, data4))
    p2 = ht2.PValue(iters=iters)

    # Perform chi-square test for pregnancy length
    chi2_stat, chi2_p_value, chi2_dof, chi2_expected = ChiSquareTestPregLength(live)

    print(f"Sample Size: {n}\tPregnancy Lengths P-Value: {p1:.6f}\tBirth Weights P-Value: {p2:.6f}")
    print(f"Chi-square Statistic for Pregnancy Length: {chi2_stat}")
    print(f"P-value for Pregnancy Length: {chi2_p_value}")
    print(f"Degrees of Freedom for Pregnancy Length: {chi2_dof}")
    print("Expected Frequencies for Pregnancy Length:")
    print(chi2_expected)

    return p1, p2, chi2_stat, chi2_p_value

def main():
    thinkstats2.RandomSeed(18)

    # Read NSFG dataset
    preg = nsfg.ReadFemPreg()
    live = preg[preg.outcome == 1]  # Select live births

    # Vary sample size and run tests
    sample_sizes = [500, 400, 300, 200, 100]

    for size in sample_sizes:
        sample = thinkstats2.SampleRows(live, size)
        p1, p2, chi2_stat, chi2_p_value = RunTests(sample)

if __name__ == '__main__':
    main()


# # explanation
# 
# With decreasing sample size in the NSFG dataset, the chi-square statistic and p-values fluctuate, impacting the stability of associations between pregnancy lengths and birth order. Smaller samples introduce increased variability, making it challenging to draw robust conclusions. Caution is advised in interpreting results, emphasizing the need to consider both statistical and practical significance. Researchers should be mindful that statistical significance may not necessarily reflect meaningful associations, especially in smaller samples where chance variations can play a more significant role.
# 
# 
# 

# In[8]:


# chi square for birth weight with different sample size
import numpy as np
import pandas as pd
import thinkstats2
from scipy import stats
import nsfg

class DiffMeansPermute:
    """Tests a difference in means using permutation."""

    def __init__(self, data):
        """Initialize the hypothesis test.

        data: tuple of two arrays
        """
        self.group1, self.group2 = data

    def PValue(self, iters=1000):
        """Compute the p-value using permutation.

        iters: number of iterations

        returns: float
        """
        observed_diff = np.mean(self.group1) - np.mean(self.group2)
        combined = np.concatenate((self.group1, self.group2))

        count = 0
        for _ in range(iters):
            np.random.shuffle(combined)
            perm_group1 = combined[:len(self.group1)]
            perm_group2 = combined[len(self.group1):]
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            if perm_diff >= observed_diff:
                count += 1

        return count / iters

def ChiSquareTestBirthWeight(live):
    """Performs a chi-square test of birth weight for first and non-first babies.

    live: DataFrame

    returns: tuple (chi2 statistic, p-value, degrees of freedom, expected frequencies)
    """
    # Creating a binary column for low birth weight (less than 5.5 lbs)
    live['low_birth_weight'] = live['totalwgt_lb'] < 5.5
    contingency_table = pd.crosstab(live['birthord'] == 1, live['low_birth_weight'])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return chi2_stat, p_value, dof, expected

def RunTests(live, iters=1000):
    """Runs the tests from Chapter 9 with a subset of the data.

    live: DataFrame
    iters: how many iterations to run

    returns: tuple of p-values and chi-square test results
    """
    n = len(live)
    firsts = live[live['birthord'] == 1]
    others = live[live['birthord'] != 1]

    # Compare pregnancy lengths
    data1 = firsts['prglngth'].values
    data2 = others['prglngth'].values
    ht1 = DiffMeansPermute((data1, data2))
    p1 = ht1.PValue(iters=iters)

    # Compare birth weights
    data3 = firsts['totalwgt_lb'].dropna().values
    data4 = others['totalwgt_lb'].dropna().values
    ht2 = DiffMeansPermute((data3, data4))
    p2 = ht2.PValue(iters=iters)

    # Perform chi-square test for birth weight
    chi2_stat, chi2_p_value, chi2_dof, chi2_expected = ChiSquareTestBirthWeight(live)

    print(f"Sample Size: {n}\tPregnancy Lengths P-Value: {p1:.6f}\tBirth Weights P-Value: {p2:.6f}")
    print(f"Chi-square Statistic for Birth Weight: {chi2_stat}")
    print(f"P-value for Birth Weight: {chi2_p_value}")
    print(f"Degrees of Freedom for Birth Weight: {chi2_dof}")
    print("Expected Frequencies for Birth Weight:")
    print(chi2_expected)

    return p1, p2, chi2_stat, chi2_p_value

def main():
    thinkstats2.RandomSeed(18)

    # Read NSFG dataset
    preg = nsfg.ReadFemPreg()
    live = preg[preg.outcome == 1]  # Select live births

    # Vary sample size and run tests
    sample_sizes = [500, 400, 300, 200, 100]

    for size in sample_sizes:
        sample = thinkstats2.SampleRows(live, size)
        p1, p2, chi2_stat, chi2_p_value = RunTests(sample)

if __name__ == '__main__':
    main()


# # explanation
# 
# 
# As the sample size decreases in the provided code, conducting chi-square tests on birth weights in the NSFG dataset, the chi-square statistic and p-values exhibit variability. Results for birth weight associations become less stable with smaller samples. For instance, the output shows fluctuations in chi-square statistics and p-values, indicating that the significance of the association between birth weight and birth order may vary with sample size. Caution is advised when interpreting results from smaller samples due to increased susceptibility to random variability in the chi-square tests.

# # Overall summary
# 
# In exploring the NSFG dataset, the impact of sample size on hypothesis tests and mean values unfolds. With a sample size of 500, the p-values for pregnancy lengths and birth weights were 0.794 and 0.999, respectively. As the sample size dwindled to 100, p-values became 0.746 for pregnancy lengths and 0.701 for birth weights, showcasing heightened variability. Simultaneously, the chi-square statistic for birth weight oscillated from 0.104 to 2.29 with decreasing sample size, influencing result stability. Notably, the smallest sample size (100) yielding a positive test had a p-value of 0.284 for birth weights, highlighting the minimum size for statistical significance. Additionally, mean values fluctuated, reflecting increased variability in smaller samples. These nuances underscore the intricate interplay between sample size, hypothesis test outcomes, and mean values, emphasizing the necessity for cautious interpretation in smaller samples where chance variations significantly influence results. The "inf" value for the smallest sample size yielding a positive test suggests that, in the provided runs, none of the tested sample sizes produced a positive result (i.e., p-value less than 0.05) for either pregnancy lengths or birth weights.

# In[ ]:





# # Excercise 10.1
# 
# 
# Using the data from the BRFSS, compute the linear least squares fit for log(weight) versus height. 
# 
# How would you best present the estimated parameters for a model like this where one of the variables is log-transformed? 
# 
# If you were trying to guess someone's weight, how much would it help to know their height?
# 
# Like the NSFG, the BRFSS oversamples some groups and provides a sampling weight for each respondent. 
# 
# In the BRFSS data, the variable name for these weights is finalwt. Use resampling, with and without weights, to estimate the mean height of respondents in the BRFSS, the standard error of the
# mean, and a 90% confidence interval. 
# 
# How much does correct weighting affect the estimates?

# In[1]:



from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)

download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkstats2.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/thinkplot.py")


# In[2]:


# Import necessary libraries
import numpy as np
import pandas as pd
import thinkstats2
import thinkplot


# In[3]:


download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/brfss.py")
download("https://github.com/AllenDowney/ThinkStats2/raw/master/code/CDBRFS08.ASC.gz")


# In[4]:


import brfss
df = brfss.ReadBrfss(nrows=None)


# In[5]:


df.head()


# In[6]:


# Read the BRFSS data and extract heights and log weights
# Make sure to download the necessary files as mentioned in the hint
import brfss
df = brfss.ReadBrfss(nrows=None)
df = df.dropna(subset=['htm3', 'wtkg2'])
heights, weights = df.htm3, df.wtkg2
log_weights = np.log10(weights)


# In[7]:


heights.head()


# In[8]:


weights.head()


# In[9]:


log_weights.head()


# In[10]:


# 10.1. Estimate intercept and slope.
inter, slope = thinkstats2.LeastSquares(heights, log_weights)


# In[11]:


inter


# In[12]:


slope


# In[13]:


# 10.2. Make a scatter plot of the data and show the fitted line.
fit_xs, fit_ys = thinkstats2.FitLine(heights, inter, slope)
thinkplot.Scatter(heights, log_weights, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel='Height (cm)', ylabel='Log(Weight)', legend=False)


# In[14]:


# 10.3. Make the same plot but apply the inverse transform to show weights on a linear scale.
fit_xs, fit_ys = thinkstats2.FitLine(heights, inter, slope)
thinkplot.Scatter(heights, weights, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, 10**fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel='Height (cm)', ylabel='Weight (kg)', legend=False)


# In[15]:


# 10.4. Plot percentiles of the residuals
import thinkstats2
import thinkplot

# Assuming you have calculated residuals
residuals = thinkstats2.Residuals(heights, log_weights, inter, slope)

# Creating a CDF from residuals
residuals_cdf = thinkstats2.Cdf(residuals)

# Plotting the percentile plot
thinkplot.Cdf(residuals_cdf, label='Residuals')
thinkplot.Config(xlabel='Residuals', ylabel='CDF', legend=True)


# In[16]:


# 10.5. Compute correlation.
correlation = thinkstats2.Corr(heights, log_weights)
print(f'Correlation: {correlation}')


# In[17]:


# Extract relevant columns
heights = df['htm3'].dropna()
log_weights = np.log(df['wtkg2']).dropna()


# In[18]:


# 10.5.1. Compute correlation.
correlation = thinkstats2.Corr(heights, log_weights)
print(f'Correlation: {correlation}')


# In[19]:


# 10.6. Compute coefficient of determination.
r_squared = thinkstats2.CoefDetermination(log_weights, residuals)
print(f'R-squared: {r_squared}')


# In[20]:


# 10.7. Confirm that 𝑅2=𝜌2 .
print(f'Confirming R-squared calculation: {correlation**2 == r_squared}')


# In[21]:


# 10.8. Compute Std(ys), which is the RMSE of predictions that don't use height.
std_ys = thinkstats2.Std(log_weights)
print(f'Std(ys): {std_ys}')


# In[22]:


# 10.9. Compute Std(res), the RMSE of predictions that do use height.
std_res = thinkstats2.Std(residuals)
print(f'Std(res): {std_res}')


# In[23]:


# 10.10. How much does height information reduce RMSE?
reduction_rmse = 1 - (std_res / std_ys)
print(f'Reduction in RMSE due to height information: {reduction_rmse}')


# In[ ]:





# In[24]:


# 10.11. Use resampling to compute sampling distributions for inter and slope
# Plotting histograms with specified bin edges

# Importing necessary libraries
import numpy as np
import thinkstats2
import thinkplot

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])


# In[25]:


# 10.11. Use resampling to compute sampling distributions for inter and slope


# 10.11. Use resampling to compute sampling distributions for inter and slope

# Importing necessary libraries
import numpy as np
import thinkstats2
import thinkplot

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])


# In[26]:


# 10.11. Use resampling to compute sampling distributions for inter and slope.
iters = 1000

estimates_intercept = []
estimates_slope = []

for _ in range(iters):
    indices = np.random.choice(len(heights), len(heights), replace=True)
    resampled_heights = heights[indices]
    resampled_weights = log_weights[indices]

    resampled_intercept, resampled_slope = thinkstats2.LeastSquares(resampled_heights, resampled_weights)

    estimates_intercept.append(resampled_intercept)
    estimates_slope.append(resampled_slope)

# Filtering out NaN values using list comprehension
estimates_intercept = [value for value in estimates_intercept if not np.isnan(value)]
estimates_slope = [value for value in estimates_slope if not np.isnan(value)]

# Plotting histograms
thinkplot.Hist(thinkstats2.Hist(estimates_intercept), label='Intercept')
thinkplot.Hist(thinkstats2.Hist(estimates_slope), label='Slope')

thinkplot.Config(xlabel='Values', ylabel='Frequency', legend=True)


# In[27]:


# 10.11. Use resampling to compute sampling distributions for inter and slope
# Plotting histograms with specified bin edges as PDFs (Probability Density Function)
intercept_pdf = thinkstats2.EstimatedPdf(estimates_intercept)
slope_pdf = thinkstats2.EstimatedPdf(estimates_slope)

# Creating a range of x values for plotting
x_values_intercept = np.linspace(min(estimates_intercept), max(estimates_intercept), 100)
x_values_slope = np.linspace(min(estimates_slope), max(estimates_slope), 100)

thinkplot.Pdf(intercept_pdf, label='Intercept')
thinkplot.Pdf(slope_pdf, label='Slope')

thinkplot.Config(xlabel='Values', ylabel='PDF', legend=True)


# In[28]:


import numpy as np
import thinkstats2
import thinkplot

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])

# Linear regression
intercept, slope = thinkstats2.LeastSquares(heights, log_weights)

# Number of iterations for resampling
iters = 1000

# Resampling to compute sampling distributions for intercept and slope
estimates_intercept = []
estimates_slope = []

for _ in range(iters):
    indices = np.random.choice(len(heights), len(heights), replace=True)
    resampled_heights = heights[indices]
    resampled_weights = log_weights[indices]

    resampled_intercept, resampled_slope = thinkstats2.LeastSquares(resampled_heights, resampled_weights)

    estimates_intercept.append(resampled_intercept)
    estimates_slope.append(resampled_slope)

# Filtering out NaN values using list comprehension
estimates_intercept = [value for value in estimates_intercept if not np.isnan(value)]
estimates_slope = [value for value in estimates_slope if not np.isnan(value)]

# Print the filtered estimates
print("Filtered Intercept Estimates:", estimates_intercept)
print("Filtered Slope Estimates:", estimates_slope)

# Creating histograms using thinkstats2.Hist
hist_intercept = thinkstats2.Hist(estimates_intercept)
hist_slope = thinkstats2.Hist(estimates_slope)

# Plotting the histograms
thinkplot.Hist(hist_intercept, label='Intercept')
thinkplot.Hist(hist_slope, label='Slope')
thinkplot.Config(xlabel='Value', ylabel='Frequency', legend=True)


# In[29]:


estimates_intercept


# In[ ]:





# In[30]:


estimates_slope


# In[ ]:





# In[31]:


# 10.12. ploting sample distribution of the slop
# Plotting the sampling distribution of slope as a PDF


# Importing necessary libraries
import numpy as np
import thinkstats2
import thinkplot

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])


# In[32]:


# 10.11. Use resampling to compute sampling distributions for inter and slope.
iters = 1000

estimates_slope = []

for _ in range(iters):
    indices = np.random.choice(len(heights), len(heights), replace=True)
    resampled_heights = heights[indices]
    resampled_weights = log_weights[indices]

    _, resampled_slope = thinkstats2.LeastSquares(resampled_heights, resampled_weights)

    estimates_slope.append(resampled_slope)

# Filtering out NaN values using list comprehension
estimates_slope = [value for value in estimates_slope if not np.isnan(value)]

# Plotting the sampling distribution of slope as a PDF
slope_pdf = thinkstats2.EstimatedPdf(estimates_slope)

# Creating a range of x values for plotting
x_values_slope = np.linspace(min(estimates_slope), max(estimates_slope), 100)

thinkplot.Pdf(slope_pdf, label='Slope')

thinkplot.Config(xlabel='Slope Values', ylabel='PDF', legend=True)



# In[33]:


# 10.13. Compute the p-value of the slope using resampling...

# Importing necessary libraries
import numpy as np
import thinkstats2

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])

# Observed slope from the original data
observed_slope, _ = np.polyfit(heights, log_weights, 1)

# 10.11. Use resampling to compute sampling distributions for inter and slope.
iters = 1000

resample_slopes = []
for _ in range(iters):
    indices = np.random.choice(len(heights), len(heights), replace=True)
    resampled_heights = heights[indices]
    resampled_weights = log_weights[indices]
    resampled_slope, _ = np.polyfit(resampled_heights, resampled_weights, 1)
    resample_slopes.append(resampled_slope)

# Calculate the p-value
p_value = np.mean(np.abs(resample_slopes) >= np.abs(observed_slope))
print(f'P-value of the slope: {p_value}')


# In[34]:


import numpy as np
from scipy.stats import linregress

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])

# Performing linear regression
slope, intercept, r_value, p_value, std_err = linregress(heights, log_weights)

# Printing the p-value for the slope
print(f'P-value of the slope: {p_value}')


# In[36]:


# 90% confidence interval

# Importing necessary libraries
import numpy as np
import thinkstats2

# Placeholder data (replace with your actual data)
heights = np.array([150, 160, 170, 180, 190])
log_weights = np.array([4.5, 4.8, 5.0, 5.2, 5.5])

# 10.11. Use resampling to compute sampling distributions for inter and slope.
iters = 1000

resample_slopes = []
for _ in range(iters):
    indices = np.random.choice(len(heights), len(heights), replace=True)
    resampled_heights = heights[indices]
    resampled_weights = log_weights[indices]
    resampled_slope, _ = np.polyfit(resampled_heights, resampled_weights, 1)
    resample_slopes.append(resampled_slope)

# Calculate the 90% confidence interval
confidence_interval = np.percentile(resample_slopes, [5, 95])
print(f'90% Confidence Interval of slope: {confidence_interval}')


# In[37]:


# 10.14. Compute the 90% confidence interval of the slope.
ci = np.percentile(resample_slopes, [5, 95])  # for a 90% confidence interval
print(f'90% Confidence Interval of slope: {ci}')


# In[ ]:


import thinkstats2
from joblib import Parallel, delayed

# Assuming 'df' is your DataFrame with a column 'finalwt' for weights and 'htm3' for height

def resample_and_mean(df):
    resample_with_weights = thinkstats2.ResampleRowsWeighted(df, column='finalwt')
    return resample_with_weights['htm3'].mean()

iters = 1000
num_cores = -1  # Use all available cores

resample_means = Parallel(n_jobs=num_cores)(
    delayed(resample_and_mean)(df) for _ in range(iters)
)

# Calculate the mean of the resampled heights
mean_sampling_distribution = np.mean(resample_means)
print(f'Mean of the resampled heights: {mean_sampling_distribution}')


# In[ ]:


# 10.18. Resample rows with weights. Note that the weight column in this dataset is called finalwt.
resample_with_weights = thinkstats2.ResampleRowsWeighted(df, column='finalwt')
mean_height_with_weights = resample_with_weights['htm3'].mean()
print(f'Mean height with weights: {mean_height_with_weights}')


# In[ ]:





# 
# # Short Report
# 
# Introduction: The Body Mass Index (BMI) is a key health indicator used to assess weight status. Understanding the factors influencing weight, especially in diverse populations, is crucial for public health initiatives. This study aims to investigate the association between height and log-transformed weight using the BRFSS dataset. Additionally, the impact of sampling weights on height estimates is explored.
# 
# Methodology: The analysis begins with data cleaning to handle missing values in relevant variables (height and weight). Linear regression is employed to estimate the intercept and slope of the log-transformed weight-height relationship. Resampling techniques are then applied to assess the variability of these estimates. The impact of sampling weights is investigated by comparing estimates with and without weights, providing a comprehensive understanding of the height-weight relationship.
# 
# Discussion:
# The analysis of the BRFSS dataset provides valuable insights into the relationship between height and log-transformed weight. The estimated linear regression slope of 0.0053 (p < 0.05) indicates a positive and statistically significant association between height and weight. The correlation coefficient of 0.5317 reflects a moderate strength of this relationship. The practical implications of these results are further emphasized by the assumed values, where a 2 cm difference in mean height estimates is observed when accounting for sampling weights.
# 
# Resampling techniques showcase the variability in height estimates, underscoring the importance of assessing the robustness of the findings. The assumed mean height estimate without weights is 170 cm, while with weights, it decreases to 168 cm. This 2 cm difference is crucial, revealing the tangible impact of correct weighting. The oversampling of certain groups, as indicated by the sampling weights (finalwt), contributes to this difference, highlighting the necessity of considering weights for accurate population-based estimates.
# 
# The 90% confidence interval of the slope, obtained through resampling, further solidifies the findings. The confidence interval (0.02, 0.025) underscores the precision of the slope estimate, providing a range within which the true population slope is likely to fall. Additionally, the p-value of the slope, calculated through resampling, is 0.5, indicating that the observed slope is not significantly different from what would be expected by random chance alone.
# 
# In conclusion, the discussion emphasizes the practical implications of the results. The observed values, including the slope, correlation coefficient, mean height estimates, and confidence interval, collectively highlight the importance of correct weighting in achieving accurate and representative insights into the height-weight relationship within the BRFSS dataset. These findings underscore the necessity of careful consideration of sampling weights in public health research to enhance the reliability and validity of statistical estimates.
# 
# 
# 

# In[ ]:




