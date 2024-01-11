# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: September 6, 2023
#
# You can solve the exercises below by using standard Python 3.10 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.10/), [NumPy](https://numpy.org/doc/stable/user/index.html), [Matplotlib](https://matplotlib.org/stable/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html), [PyMC](https://docs.pymc.io).
# You can also look at the [slides of the course](https://homes.di.unimi.it/monga/lucidi2223/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
# **It is forbidden to communicate with others.**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm   # type: ignore
import arviz as az   # type: ignore

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

data = pd.read_csv('brown_bear_blood.csv', parse_dates=['birth', 'sampling_date'])
data.head()

# ### Exercise 2 (max 3 points)
#
# Add a column `age_days` with the exact number of days between `birth` and `sampling_date`. The column should have dtype `int64`.
#

data['age_days'] = (data['sampling_date'] - data['birth']).dt.days
data['age_days'].dtype


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

def correct_age(sex: str, env: str, age: int) -> float:
    """Correct age according to sex and environment.
    
    >>> np.isclose(correct_age('M', 'wild', 100), 80) 
    True
    
    """
    
    assert sex == 'M' or sex == 'F', f'invalid sex: {sex}'
    assert env == 'wild' or env == 'captive', f'invalid environment: {env}'
    assert age > 0, f'Age should be positive'
    
    if sex == 'M':
        if env == 'wild':
            return 0.8*age
        else:
            return 1.*age
    else:
        if env == 'wild':
            return 1.2*age
    return 1.5*age


# +
# You can test your docstrings by uncommenting the following two lines

import doctest
doctest.testmod()
# -

# ### Exercise 4 (max 4 points)
#
# Apply the function defined in Exercise 3 on the bears at least 60 days old (at sampling date).

data[data['age_days'] >= 60].apply(lambda r: correct_age(r['sex'], r['environment'], r['age_days']), axis=1)

# ### Exercise 5 (max 4 points)
#
# Each `Sample_ID` is composed by a date and a site name. Print all the unique names of the sites together with the number of samples collected in that site.

data['Sample_ID'].str.split().apply(lambda lst: lst[1]).value_counts().sort_index()

# ### Exercise 6 (max 4 points)
#
# Plot together the histograms of `age_days` for each combination of sex and environment. The four histograms should appear within the same axes.

data[(data['sex'] == 'M') & (data['environment'] == 'wild')]['age_days'].describe()

fig, ax = plt.subplots(1)
for s in ('M', 'F'):
    for e in ('wild', 'captive'):
        ax.hist(data[(data['sex'] == s) & (data['environment'] == e)]['age_days'], 
                bins='auto', alpha=.6, label=f'{s} {e}')
_ = ax.legend()



# ### Exercise 7 (max 5 points)
#
# Make a figure with 2 columns and 4 rows. In the first column put the scatter plots of `age_days` vs. the four methylation levels (`SLC12A5`,`POU4F2`,`VGF`,`SCGN`), in the second column the scatter plots of the ages corrected according to the function defined in Exercise 3 vs. the four methylation levels.

data['corrected_age'] = data.apply(lambda r: correct_age(r['sex'], r['environment'], r['age_days']), axis=1)

# +
fig, ax  = plt.subplots(nrows=4, ncols=2)
for i, m in enumerate(('SLC12A5','POU4F2','VGF','SCGN')):
    ax[i,0].scatter(data['age_days'], data[m])
    ax[i,1].scatter(data['corrected_age'], data[m])
    ax[i,0].set_title(f'age vs. {m}')
    ax[i,1].set_title(f'corrected age vs. {m}')

    
fig.tight_layout()
# -

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
    a = pm.Normal('alpha', 0, 1)
    b = pm.Normal('beta', 0, 1)
    g = pm.Exponential('gamma', 1)
    
    pm.Normal('age_days', sigma=g, mu=a + b*data['SLC12A5'], observed=data['age_days'])

with model:
    idata = pm.sample(random_seed=328432)

_ = az.plot_posterior(idata)


