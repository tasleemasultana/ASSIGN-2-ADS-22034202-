#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats


# In[18]:


def choose_data(filename):
    ''' selects data from the dataset
        and returns the selected data 
    '''
    path = r"C:\Users\Jalaluddin Shaik\OneDrive\Desktop"
    API_Data = pd.read_excel(path + "/" + filename)
    API_Data_Archive = API_Data.copy()
    API_Data1 = API_Data[2:]
    API_Data1.columns = API_Data1.iloc[0]
    API_Data = API_Data1[1:]
    
    API_Data2 = pd.melt(API_Data, id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                        var_name = "Year", value_name = "Value")
    
  
    
    Country_As_Columns = API_Data2.pivot(index = ['Indicator Name', 'Indicator Code', 'Year'], columns = 'Country Name', 
                                      values = 'Value').reset_index()
    
    Date_As_Columns = API_Data2.pivot(index = ['Country Name', 'Country Code', 'Indicator Name'], columns = 'Year', 
                                      values = 'Value').reset_index()
    return Date_As_Columns , Country_As_Columns


# In[19]:


Date_As_Columns, Country_As_Columns = choose_data("API_19_DS2_en_excel_v2_5360124.xlsx")


# In[20]:


Date_As_Columns['Indicator Name'].unique()


# In[21]:


Date_As_Columns.describe()


# In[22]:


Country_As_Columns.describe()


# In[23]:


Urban_pop = Date_As_Columns[(Date_As_Columns['Indicator Name'] == 'Urban population growth (annual %)') & (Date_As_Columns['Country Name'] == 'India')]
Urban_pop.describe()


# In[24]:


Dates_Country_choosen = Date_As_Columns[(Date_As_Columns['Country Name'] == 'India')|(Date_As_Columns['Country Name'] == 'Australia')
                |(Date_As_Columns['Country Name'] == 'Japan')|(Date_As_Columns['Country Code'] == 'USA')
                |(Date_As_Columns['Country Name'] == 'United Kingdom')|(Date_As_Columns['Country Name'] == 'Germany')
                |(Date_As_Columns['Country Name'] == 'China')|(Date_As_Columns['Country Name'] == 'Brazil')]
API_Data3 = pd.melt(Dates_Country_choosen, id_vars = ['Country Name', 'Country Code', 'Indicator Name'], 
                        var_name = "Year", value_name = "Value")
API_Data4 = API_Data3[API_Data3['Indicator Name'] == 'CO2 emissions (kt)']


# In[25]:


Country_Name = API_Data4['Country Name'].unique()
for i in Country_Name:
#     plt.figure(figsize=(17,6))
    country_i = API_Data4[API_Data4['Country Name'] == i].sort_values('Year')
    plt.plot(country_i['Year'], country_i['Value'], label = i)
    plt.title('CO2 Emission', size = 15)
    plt.xlabel('Year', size = 15)
    plt.ylabel('CO2 emission(kt)', size = 15)
    plt.legend()
    plt.plot()
plt.show()
    


# In[26]:


API_Data5 = API_Data3[API_Data3['Indicator Name'] == 'Electric power consumption (kWh per capita)']
Country_Name = API_Data5['Country Name'].unique()
for i in Country_Name:
#     plt.figure(figsize=(17,6))
    country_i = API_Data5[API_Data5['Country Name'] == i].sort_values('Year')
    plt.plot(country_i['Year'], country_i['Value'], label = i)
    plt.title('Power Consumption', size = 15)
    plt.xlabel('Year', size = 15)
    plt.ylabel('Power consumption(kWh/capita)', size = 15)
    plt.legend()
    plt.plot()
plt.show()


# In[27]:


API_Data6 = API_Data3[API_Data3['Indicator Name'] == 'Population, total']
Country_Name = API_Data6['Country Name'].unique() 

for i in Country_Name:
#     plt.figure(figsize=(17,6))
    country_i = API_Data6[API_Data6['Country Name'] == i].sort_values('Year')
    plt.plot(country_i['Year'], country_i['Value'], label = i)
    plt.title('Total Population', size = 15)
    plt.xlabel('Year', size = 15)
    plt.ylabel('Population', size = 15)
    plt.legend()
    plt.plot()
plt.show()


# In[28]:


median_CO2emission = np.median(API_Data6['Value'])
median_CO2emission


# In[29]:


aver = np.mean(API_Data6["Value"])
print("Average:", aver)
std = np.std(API_Data6["Value"])
print("Std. deviation:", std)
print("Skew:", stats.skew(API_Data6["Value"]))
print("Kurtosis", stats.kurtosis(API_Data6["Value"]))


# In[30]:


API_Data3


# In[31]:


x = [1991, 1994, 1997, 2000, 2003, 2006, 2009, 2012, 2015, 2018]
index = np.arange(len(x))
China = API_Data3[(API_Data3['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')&
                      (API_Data3['Country Name'] == 'China')&(API_Data3['Year'].isin(x))]
USA = API_Data3[(API_Data3['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')&
                      (API_Data3['Country Code'] == 'USA')&(API_Data3['Year'].isin(x))]
bar_width = 0.35

fig, ax = plt.subplots()
ax.bar(index, China['Value'], bar_width, label = 'China')
ax.bar(index+bar_width, USA['Value'], bar_width, label = 'USA')
ax.set_title('% Renewable Energy Consumption', size = 15)
ax.set_xlabel('Year', size = 15)
ax.set_ylabel('% Renewable energy', size = 15)
tick_positions = np.arange(0, len(x), 1)
tick_labels = x
ax.set_xticks(tick_positions + bar_width / 2)
ax.set_xticklabels(tick_labels)
ax.legend()
plt.show()


# In[32]:


API_Data8 = API_Data3[(API_Data3['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')|
         (API_Data3['Indicator Name'] == 'Electric power consumption (kWh per capita)')|
         (API_Data3['Indicator Name'] == 'Population, total')|
         (API_Data3['Indicator Name'] == 'CO2 emissions (kt)')|
         (API_Data3['Indicator Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)')]
pivotdata = API_Data8.pivot(index = ['Country Name', 'Year', 'Country Code'], columns = 'Indicator Name', 
                            values = 'Value').reset_index()
data_USA = pivotdata[pivotdata['Country Code'] == 'USA']

# Create a dictionary to rename the columns
col_names = {'Renewable energy consumption (% of total final energy consumption)' : '%Renewable Energy',
             'Electric power consumption (kWh per capita)' : 'Power Consumption(kWh/capita)',
             'Population, total' : 'Population',
             'CO2 emissions (kt)' : 'CO2 Emissions (kt)',
             'Total greenhouse gas emissions (kt of CO2 equivalent)' : 'Total GHG Emissions(kt CO2 eq)'}

# Rename the columns
data_USA = data_USA.rename(columns = col_names)

# Compute the correlation matrix and create the heat map
corr2 = data_USA.corr()
heat2 = sns.heatmap(corr2)
plt.title('USA', size = 15)

plt.show()

