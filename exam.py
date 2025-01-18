# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: September 6, 2023
#
# You can solve the exercises below by using standard Python 3.11 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.11/), [NumPy](https://numpy.org/doc/stable/user/index.html), [Matplotlib](https://matplotlib.org/stable/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html), [PyMC](https://docs.pymc.io).
# You can also look at the slides or your code on [GitHub](https://github.com). 
#
# **It is forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question)**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#

import numpy as np
import pandas as pd             # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm               # type: ignore
import arviz as az              # type: ignore

# ### Exercise 1 (max 3 points)
#
# The file [brown_bear_blood.csv](./brown_bear_blood.csv) (Shimozuru, Michito, Nakamura, Shiori, Yamazaki, Jumpei, Matsumoto, Naoya, Inoue-Murayama, Miho, Qi, Huiyuan, Yamanaka, Masami, Nakanishi, Masanao, Yanagawa, Yojiro, Sashika, Mariko, Tsubota, Toshio, & Ito, Hideyuki. (2023). *Age estimation based on blood DNA methylation levels in brown bears* https://doi.org/10.5061/dryad.9w0vt4bm0) contains
#
#  - Bear ID
#  - Birth date. It was assumed that all bears were born on February 1.
#  - Date of the blood sampling.
#  - Ages were determined at the time of blood sampling
#  - Sex, F: female, M: male.
#  - Growth environment (i.e., captive or wild).
#  - Values of the methylation levels of the samples. As PCR for each sample was conducted in duplicate, the average  value was taken as the methylation level for each sample. 
#
# Load the data in a Pandas dataframe. Be sure the columns with dates have the correct dtype (`datetime64[ns]`) and the dates are parsed correctly (the birth date is always on February 1).

# Load the data ensuring that the dates have the right dtype with parse_dates:
#I parse each object in 'birth' and 'sampling_date' as date

df = pd.read_csv('brown_bear_blood.csv', parse_dates=['birth', 'sampling_date'])

# Display the first few rows
print(df.head())

# ### Exercise 2 (max 3 points)
#
# Add a column `age_days` with the exact number of days between `birth` and `sampling_date`. The column should have dtype `int64`.
#

#Now that dates have the correct data type, i can subtract them and return the resul in number of days
df['age_days'] = (df['sampling_date'] - df['birth']).dt.days

# ### Exercise 3 (max 5 points)
#
# Define a function `correct_age` that takes a sex (F or M), an environment (wild or captive) and and age (in days), then it returns an age corrected by a factor (i.e., multiplied by) according to this table.
#
# | sex         | environment     | age correction |
# |--------------|-----------|------------|
# | M | wild     | 0.8       |
# | M | captive  | 1         |
# | F | wild     | 1.2       |
# | F | captive  | 1.5       |
#
# For example, a wild male bear with an age of 100 days, should get a corrected age of 80. 
#
#
# To get the full marks, you should declare correctly the type hints and add a test within a doctest string.

def correct_age(sex: str, environment: str, age: int) -> float:#Here all types are defined
    #Here is reported the doctest
    """
    Correct the age based on sex and environment.

    Args:
    sex (str): 'F' for female, 'M' for male.
    environment (str): 'wild' or 'captive'.
    age (int): Age in days.

    Returns:
    float: Corrected age.

    >>> correct_age('M', 'wild', 100)
    80.0
    >>> correct_age('F', 'captive', 100)
    150.0
    """
    #I create a dictionary that associates each combination of environment and sex to a correction factor
    
    correction_factors = {
        ('M', 'wild'): 0.8,
        ('M', 'captive'): 1.0,
        ('F', 'wild'): 1.2,
        ('F', 'captive'): 1.5
    }
    
    #I return the age multiplied by the correction factor
    
    return age * correction_factors[(sex, environment)]

#NB! This function applies to single values of sex, environment and age

# +
# You can test your docstrings by uncommenting the following two lines

#import doctest
#doctest.testmod()
# -

# ### Exercise 4 (max 4 points)
#
# Apply the function defined in Exercise 3 on the bears at least 60 days old (at sampling date).

#1-To apply the function to rows in all the dataframe I need to use df.apply
#2-The lambda function takes row (each row of the dataframe) as an argument: 
#  it verifies if the age in days of the bear is higher or equal to 60 (if statement)
#  if the bear is older or 60 days old, the value added to the new column will be the
#  value returned by the correct_age function, otherwise it will be the original age_days
#3-axis=1 applies the function from the first row, the first to actually contain data

df['corrected_age'] = df.apply(
    lambda row: correct_age(row['sex'], row['environment'], row['age_days']) if row['age_days'] >= 60 else row['age_days'],
    axis=1
)

# ### Exercise 5 (max 4 points)
#
# Each `Sample_ID` is composed by a date and a site name. Print all the unique names of the sites together with the number of samples collected in that site.

#1-I create the 'Site' column and apply a lamda function that splits
#  'Sample_ID' in 2
#2-I consider only the second elemnt of the split, in this case the site
#3-I evaluate the number of elements in the dataframe with the same site

df['Site'] = df['Sample_ID'].apply(lambda x: x.split(' ')[1])
site_counts = df['Site'].value_counts()
print(site_counts)

# ### Exercise 6 (max 4 points)
#
# Plot together the histograms of `age_days` for each combination of sex and environment. The four histograms should appear within the same axes.

#1-I use a for loop to group the dataframe by the specified columns:
#  in particular i'm creating different groups (subsets), each with a unique
#  combination of sex and environment
#2-I plot an histogram for each group, containing age_days of the individuals

fig, ax = plt.subplots()
for (sex, env), group in df.groupby(['sex', 'environment']):
    group['age_days'].plot(kind='hist', ax=ax, alpha=0.5, label=f'{sex} - {env}')
ax.legend()
plt.show()


# ### Exercise 7 (max 5 points)
#
# Make a figure with 2 columns and 4 rows. In the first column put the scatter plots of `age_days` vs. the four methylation levels (`SLC12A5`,`POU4F2`,`VGF`,`SCGN`), in the second column the scatter plots of the ages corrected according to the function defined in Exercise 3 vs. the four methylation levels.

#1-The enumerate function is used to get both the index (i)
#  and the value (methylation) of each element in the listù
#2-I create a plot for each methylation level:
#       - one for age_days
#       - one for corrected_age

fig, axes = plt.subplots(4, 2, figsize=(10, 20))

methylation_levels = ['SLC12A5', 'POU4F2', 'VGF', 'SCGN']
for i, methylation in enumerate(methylation_levels):
    df.plot.scatter(x='age_days', y=methylation, ax=axes[i, 0])
    df.plot.scatter(x='corrected_age', y=methylation, ax=axes[i, 1])

plt.tight_layout()
plt.show()



# 

# ### Exercise 8 (max 5 points)
#
# Consider this statistical model:
#
# - a parameter $\alpha$ is normally distributed with $\mu = 0$ and $\sigma = 1$ 
# - a parameter $\beta$ is normally distributed with $\mu = 1$ and $\sigma = 1$ 
# - a parameter $\gamma$ is exponentially distributed with $\lambda = 1$
# - the observed `age_days` is normally distributed with standard deviation $\gamma$ and a mean given by $\alpha + \beta \cdot M$ (where $M$ is the correspondig value of `SLC12A5`).
#
# Code this model with pymc, sample the model, and plot the summary of the resulting estimation by using `az.plot_posterior`. 
#
#
#

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=1, sigma=1)
    gamma = pm.Exponential('gamma', lam=1)
    
    mu = alpha + beta * df['SLC12A5']
    age_obs = pm.Normal('age_obs', mu=mu, sigma=gamma, observed=df['age_days'])
    
    trace = pm.sample(1000, tune=1000)
    
az.plot_posterior(trace)
plt.show()
