#!/usr/bin/env python
# coding: utf-8

# # **ONLINE SHOPPER'S INTENTION ANALYSIS**
# 
# 
# 
# **BY COURAGE SIAMEH.**

# **INTRODUCTION**
# 
# Online shopping by consumers is increasing every year, but the conversion rates have stayed relatively stable. For instance, many of us explore e-commerce platforms like Amazon, may add things to our wishlists or shopping carts, but ultimately make no purchases. This reality highlights the necessity for tools and strategies that can tailor promotions and ads to online shoppers and enhance conversion rates. This project will explore multiple factors that influence a buyer's decision.
# 
# **DATASET**
# 
# We will be utilizing information from the Online Shoppers Purchasing Intention Dataset for this project, which is accessible through the UCI repository. The primary dataset can be located at this link: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset.
# 
# 
# **PROJECT MOTIVATION**
# 
# The motivation behind this project is to address the issue of low conversion rates in e-commerce websites despite the increasing trend of online shopping. The project aims to use data from the Online Shoppers Purchasing Intention Dataset to analyze various factors that influence a purchaser's decision and explore solutions to improve conversion rates. By customizing promotions and advertisements for online shoppers based on their behavior, preferences, and characteristics, the project aims to improve the overall shopping experience and increase sales for e-commerce websites.
# 

# ***Importing various Python Libraries***

# In[38]:


#importing all packages 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ***Importing The Dataset***

# In[39]:


#importing the dataset into the dataframe df 
df = pd.read_csv('online_shoppers_intention.csv')


# ***Data Assessment and Wrangling***

# In[40]:


#assessing the first 14 rows of our dataset
df.head(14)


# In[41]:


# Getting information about features in the dataframe
df.info()


# In[42]:


# checking for null values here
df.isnull().sum() 


# So it appears we have no null values in our dataframe, so we continue with our analysis.

# ## Exploratory Data Analysis
# This very part of the project will take a majority of out time as it is the inception of investigating the data to find hidden outliers and patterns. We go further by plotting them into a visualization.
# The analysis techniques employed in the EDA takes three parts following as;
# * Univariate Analysis
#     
# * Bivariate Analysis
# 
# * Linear Relationship
# 
# In neach phase of these techniques, various corresponding columns would be explored since not all the columns features can be explored by all three analysis techniques.
# 
# 
# 
#   

# # Univariate Analysis
# Each feature or column of the dataframe is analyzed here to uncover the distribution or pattern of data. We begin analysing each of the following features in detail;
# * Revenue
# * Visitor type
# * Traffic type
# * Region
# * Weekend-wise distribution
# * Browser and Operating system
# * Special day
# 
# We begin analysing each of the features above in detail to gain more insight on what we're at.

# In[43]:


# looking at the summary statistics of the data
df.describe().style.background_gradient(cmap = 'winter_r')


# ### Baseline Conversion (Rate From The [Revenue] Column)
# Here, we're looking at the number of online shopping sessions that ended in a purchase(s). The conversion rate is then calculated.

# In[44]:


#dtype of the Revenue feature
df.Revenue.dtype


# In[45]:


# visualizing a countplot of the revenue column
sns.countplot(data = df, x = 'Revenue')
plt.title("Baseline Revenue Conversion",fontsize = 15)
plt.show()


# The preceeding countplot shows **False** having a higher count compared to **True**.                                
# Remember the [Revenue] column is of a boolean dtype.

# In[46]:


# the value counts of each subcategory in our feature, the exact values needed for calculating the conversion rate
print(df['Revenue'].value_counts())
print()
print(df['Revenue'].value_counts(normalize = True))


# From the preceding data, a total of 1908 ended up making a purchase, while 10422 did not make any.
# The conversion rate of online visitors versus overall visitors is the ratio between the total number of online sessions that led to a purchase divided by the total number of sessions. This is calculated as:

# In[47]:


1908 / 12330 * 100


# With 12,330 depicting the overall number of visitors, the conversion rate calculated was 15.47%

# In[ ]:





# ### Visitor - Wise Distribution
# We further our analysis by looking at the distribution of visitors to the website, to determine the visitor type that is most frequent. It is determined whether they are new visitors, returning visitors or visitors of any other category.

# In[48]:


# visualizing a countplot of the VisitorType column
sns.countplot(data = df, x = 'VisitorType')
plt.title("Visitor Type Distribution of Our Online Shoppers",fontsize = 15)
plt.show()


# In[49]:


#looking at the value counts of each visitor type 
print(df['VisitorType'].value_counts())
print()

"""Setting the normalize parameter to True normalizes the counts to proportions or percentages, 
such that the output shows the relative frequency of each unique value as a percentage of the total number of observations 
in the VisitorType column."""

print(df['VisitorType'].value_counts(normalize = True))


# The preceding data shows a higher number of returning visitors compared to that of new visitors. It is quite safe to assume 
# there's much success in attracting customers back to the website as shown in the countplot too.

# In[ ]:





# ### Traffic-Wise Distribution
# Considering this feature, we want to find out just how the visitors visit our webpage to help
# 1. Determine the amount of site trafffic accounted for by direct visitors
# 2. how much is generated through other mediums, such as blogs, advertisements, to mention a few.

# In[50]:


# visualizing a countplot of the TrafficType column
sns.countplot(data = df, x = 'TrafficType')
plt.title("Traffic-Wise Distribution",fontsize = 15)
plt.show()


# Type 2 appers to have the highest count amongst all the types.

# In[51]:


"""the line of code below counts the number of occurrences of each unique value in the 'TrafficType' column. 
The normalize parameter is set to True which returns the relative frequencies of each unique value instead
of the absolute counts"""

df.TrafficType.value_counts(normalize = True)


# The preceding result is evident enough that sources 2,1,3 and 4 respectively accounted for majority of our web traffic.

# In[ ]:





# ### Analysing The Distribution Of Customers Session Online
# This part of our analysis takes the distribution of customers over the days of the week to determine whether customers are more 
# or less active during the weekdays or weekens. 

# In[52]:


# visualizing a countplot of the Weekend column
sns.countplot(data = df, x = 'Weekend')
plt.title("Weekend Distribution of Our Online Shoppers",fontsize = 15)
plt.show()


# Looks like we've got more customers active on the weekdays compared to the weekends.

# In[53]:


# the value counts of each subcategory in our feature
print(df['Weekend'].value_counts())
print()
print(df['Weekend'].value_counts(normalize = True))


# The data shows that out of the 12330 online visits by customers to the website,
# 
#     9462 active customers visit the website on Weekdays     
#     2868 active customers visit the website on Weekends     
# More visitors visit the website during weekdays than weekends

# In[ ]:





# ### Region - Wise Distribution 
# Region - Wise Distribution analysis is to find out which region has the highest number of visitors to out shopping website.
# 

# In[54]:


# visualizing a countplot of the Region column
sns.countplot(data = df, x = 'Region')
plt.title("Region-Wise Distribution Of Our Online Shoppers",fontsize = 15)
plt.show()


# Our source data had numbers represennting the different diverse regions our customers are access our website from.
# From the graph above, Region 1 is seen to have the highest numbers of customer visits to our website, that of Region 3 and 4 are pretty high too. 

# In[55]:


# the value counts of each subcategory in our feature
print(df['Region'].value_counts())
print()
print(df['Region'].value_counts(normalize = True))


# Our most potential customers in Region 1 and 3 collectively accounted for 50% of customer online sessions. These two regions are likely going to be the best targets for our marketing camppaigns.

# In[ ]:





# ### Analysing The Browser And Operating Systems Of Customers
# The aim of this analysis is to aid in configuring our website to make it more responsive, reliable and user-friendly across multiple browser softwares and Operating Systems for our dear customers.

# ### The Browser Type

# In[56]:


# visualizing a countplot of the Browser column
sns.countplot(data = df, x = 'Browser')
plt.title("Browser-wise Distribution Of Our Online Shoppers",fontsize = 15)
plt.show()


# The browser type 2, emerges with the highest count of users thereby contributing the most to web trraffic on our site. 

# In[57]:


# the value counts of each subcategory in our feature
print(df['Browser'].value_counts())
print()
print(df['Browser'].value_counts(normalize = True))


# ### The Operating System Type of Customers

# In[58]:


# visualizing a countplot of the Browser column
sns.countplot(data = df, x = 'OperatingSystems')
plt.title("Distribution Of Our Online Shoppers Operating System Types",fontsize = 15)
plt.show()


# The Operating Sysytem tpe 2 has the highest count, thus contributing the most to website traffic.

# In[59]:


# the value counts of each subcategory in our feature
print(df['OperatingSystems'].value_counts())
print()
print(df['OperatingSystems'].value_counts(normalize = True))


# Fron the precedinng information, the OperatingSystem types, 2, 1, and 3 respectively, contribute most to website traffic.

# In[ ]:





# ### Distribution Of Customer Website Visits On Special Days 
# This session analysis the number of visitors we have on our website during special days. 
# We would like to know whether special days like National/public holidays, Valentines days, Festive seasons, affects the number of customers that visit our website. 

# In[61]:


# visualizing a countplot of the SpecialDay column
sns.countplot(data = df, x = 'SpecialDay')
plt.title("Special Days Session Distribution Of Our Customers",fontsize = 15)
plt.show()


# Special days clearly have no efffect or impact on the number of customer turnouts on our website.

# In[67]:


# percentage distribution for special days 
print(df['SpecialDay'].value_counts(normalize = True))


# In[ ]:





# ### BIVARIATE ANALYSIS
# We perform this analysis between two variables to look at their relationship. in this session er going to perform bivariate analysis between the revenue column and the following features.
# * Visitor type 
# * Region
# * Month
# * Browser type
# * Month
# * Traffic type
# * Special day

# ### Analysing Revenue Versus TrafficType
# We visualizing the rrelationship between revenue and traffic type on a countplot to give us the number of users in each traffic type and whether or not they made a purchase.

# In[63]:


# plotting Revenue and TrafficType
sns.countplot(x = "TrafficType", hue = "Revenue", data = df)
plt.legend(loc = 'right')
plt.title('TrafficType versus Revenue')
plt.show()


# From the plot, source 2, 1 and 3 respectively has more revenue conversion generated from web traffic. There are other sources with with considerate amount of web traffic and a very low revenue conversion compared to others.

# ### Analysing Revenue Versus VisitorType
# The categorical plot between Revenue and VisitorType will give us te number of customers in each subcategory, and whether or nnot they made a purchase. The values of the revenue column is of boolean dtype, the plot will define customers who make a purchase as **True**, and those who did not as **False**

# In[70]:


#categorical plot between Revenue and VisitorType
ax = sns.catplot(x="Revenue", col="VisitorType", col_wrap=3, kind="count", height=5, aspect=1, data=df)
plt.show()


# The results from the categorical plot reveals there's more revenue conversions for returning customers as compared to new customers. With this info, we can direct our focuses more to increasing new customers engagements with our website to increase overall purchases, revenue generation and customers as a whole.

# In[ ]:





# ### Analysing Revenue Versus Month

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




