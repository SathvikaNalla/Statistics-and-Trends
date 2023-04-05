import pandas as pd

def co2_emissions(filename):
    """
    This function takes a filename as input and returns two dataframes:
    A dataframe with country names as row index and years as column names.
      
    """
    
    # Load the data into a DataFrame and skip the first 4 rows
    co2_df = pd.read_csv(filename, skiprows=4)
    
    # Extract data for the years 1960 - 2021
    years_df = co2_df.loc[:, 'Country Name':'2021']
    years_df.columns = [col if not col.isdigit() else str(col) for col in years_df.columns]
    
    # Transpose the DataFrame to make countries as row index and years as columns
    countries_df = years_df.transpose()
    
    # Fill any missing values with 0
    countries_df = countries_df.fillna(0)
    
    # Set the column names for the countries and set the index name as 'Year'
    countries_df.columns = countries_df.iloc[0]
    countries_df = countries_df.iloc[1:]
    countries_df.index.name = 'Year'
    
    # Rename the 'Country Name' column as 'Year' and set it as the index for years_df
    years_df = years_df.rename(columns={'Country Name': 'Year'})
    years_df = years_df.set_index('Year')
    
    return years_df, countries_df



years_df, countries_df = co2_emissions('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv')

print(years_df)

print(countries_df)

# DISPLAYING THE TOP 8 COUNTRIES
import pandas as pd

# Load the csv file into a pandas dataframe
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', header=2)

# Filter the dataframe to only include data from the year 2019
df_2019 = df[['Country Name', '2019']]

# Sort the data by CO2 emissions in descending order
df_2019_sorted = df_2019.sort_values('2019', ascending=False)

# Select the top 50 countries with the highest CO2 emissions in 2019
top_50_countries = df_2019_sorted[:50]

# Display the top 50 countries
top_50_countries

# GETTING A DESCRIPTION OF THE DATA
countries_df.describe()

years_df.describe()

# THE GROUPED BAR CHART
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', skiprows=4)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(["China", "United States","India", "Russian Federation", "Japan", "Germany", "Indonesia", "Korea, Rep"]), ['Country Name', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']]


# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Set the figure size and create a new subplot
plt.figure(figsize=(10, 6))
ax = plt.subplot()

# Set the years and the number of bars per group
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
n_bars = len(years)

# Set the bar width and the offset between the groups
bar_width = 0.8 / n_bars
offset = bar_width / 2

# Set the colors for each year
colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

# Set the x ticks to be the country names
x_ticks = df_countries.index

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar([j + offset + bar_width*i for j in range(len(x_ticks))], df_countries[year], width=bar_width, label=year, color=colors[i])

# Set the axis labels and title
ax.set_xlabel('Country')
ax.set_ylabel('CO2 emissions (kg per PPP $ of GDP)')
ax.set_title('CO2 Emissions by Country and Year')

# Set the x ticks and labels
ax.set_xticks([j + 0.4 for j in range(len(x_ticks))])
ax.set_xticklabels(x_ticks, rotation=60)

# Add a legend
ax.legend()

# Show the plot
plt.show()

# THE GROUPED BAR CHART FOR ELECTRIC POWER CONSUMPTION
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5346386.csv', skiprows=4)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(["China", "United States","India", "Russian Federation", "Japan", "Germany", "Indonesia", "Korea, Rep"]), ['Country Name', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']]


# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Set the figure size and create a new subplot
plt.figure(figsize=(10, 6))
ax = plt.subplot()

# Set the years and the number of bars per group
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
n_bars = len(years)

# Set the bar width and the offset between the groups
bar_width = 0.8 / n_bars
offset = bar_width / 2

# Set the colors for each year
colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

# Set the x ticks to be the country names
x_ticks = df_countries.index

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar([j + offset + bar_width*i for j in range(len(x_ticks))], df_countries[year], width=bar_width, label=year, color=colors[i])

# Set the axis labels and title
ax.set_xlabel('Country')
ax.set_ylabel('Electric power consumption (kWh per capita)')
ax.set_title('Electric Power Consumption per Capita')

# Set the x ticks and labels
ax.set_xticks([j + 0.4 for j in range(len(x_ticks))])
ax.set_xticklabels(x_ticks, rotation=60)

# Add a legend
ax.legend()

# Show the plot
plt.show()

# THE LINE PLOT
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV 
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', skiprows=4)

# Select the relevant columns and rows
years = [str(year) for year in range(2010, 2020)]
countries = ["China", "United States","India", "Russian Federation", "Japan", "Germany", "Indonesia", "Korea, Rep."]
df = df.loc[df['Country Name'].isin(countries), ['Country Name'] + years]

# Pivot the DataFrame to create a line plot
df = df.set_index('Country Name').T
df.index.name = 'Year'
df.columns.name = 'Country'
ax = df.plot(figsize=(10, 6), linewidth=2.5, colormap='tab10')

# Set the title, axis labels, and legend
plt.title('CO2 Emissions by Country (2010-2019)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kt)')
plt.legend(title='Country', loc='upper left')

# Display the plot
plt.show()


# THE BOX PLOT
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv", skiprows=4)

# Select the desired countries and years
countries = ["China", "United States","India", "Russian Federation", "Japan", "Germany", "Indonesia", "Korea, Rep."]
years = [str(i) for i in range(2010, 2020)]
df = df.loc[df["Country Name"].isin(countries), ["Country Name"] + years]

# Melt the dataframe into long format for plotting
df_melt = pd.melt(df, id_vars=["Country Name"], value_vars=years, var_name="Year", value_name="CO2 Emissions (kt)")

# Generate the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df_melt.loc[df_melt["Country Name"] == country, "CO2 Emissions (kt)"] for country in countries])
plt.xticks(range(1, len(countries)+1), countries, rotation=45)
plt.title("CO2 Emissions by Country (2010-2019)")
plt.xlabel("Country")
plt.ylabel("CO2 Emissions (kt)")
plt.show()

#THE HEATMAP
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file and load the data into a pandas DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', skiprows=4)

# select the columns for the countries and the years 2010 to 2019
cols = ['Country Name'] + [str(year) for year in range(2010, 2020)]
df = df[cols]

# set the 'Country Name' column as the index
df.set_index('Country Name', inplace=True)

# select the eight countries of your choice
countries = ["China", "United States","India", "Russian Federation", "Japan", "Germany", "Indonesia", "Korea, Rep."]
df = df.loc[countries]

# create a heatmap using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(df.values, cmap='Reds')

# set the x-axis labels to the years
ax.set_xticks(range(10))
ax.set_xticklabels(range(2010, 2020))

# set the y-axis labels to the country names
ax.set_yticks(range(len(countries)))
ax.set_yticklabels(countries)

# add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('CO2 Emissions (kt)', rotation=-90, va='bottom')

# set the title of the plot
ax.set_title('CO2 Emissions Heatmap (2010-2019)')

# show the plot
plt.show()


# THE CORRELATION ANALYSIS
import pandas as pd

# set the name of the country you want to select
country_name = "United States"

# set the filenames for the four CSV files
filename1 = "API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5346386.csv"
filename2 = "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv"


# load the data for the first three CSV files
df1 = pd.read_csv(filename1, skiprows=4, usecols=['Country Name','2014','2013', '2012','2011','2010'])
df2 = pd.read_csv(filename2, skiprows=4, usecols=['Country Name','2014','2013', '2012','2011','2010'])
# concatenate the data 
df_combined = pd.concat([df1, df2,], ignore_index=True)

# select the rows for the chosen country and years
df_selected = df_combined.loc[df_combined['Country Name'] == country_name, ['2014','2013','2012','2011','2010']]

# set the row index to be the years
df_selected.index = ['TotalArableLand','NonRenewableElectricity']

# print the resulting dataframe
df_selected

df_transposed = df_selected.transpose()

# Set the column names
df_transposed.columns =  ['TotalArableLand', 'NonRenewableElectricity']

# Reset the index to make the years as rows
df_transposed = df_transposed.reset_index()

# Rename the 'index' column to 'Year'
df_transposed = df_transposed.rename(columns={'index': 'Year'})

df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')

df_transposed

# Correlation Analysis for the United States
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the CSV file as a Pandas dataframe
df = df_transposed

# remove the first column ('Year') since we don't want to plot it
df = df.iloc[:, 1:]

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix as a heatmap with correlation values displayed inside each box
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)

# loop over each box and display the correlation value as text inside the box
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2), ha='center', va='center', color='w')

plt.colorbar(im)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', skiprows=4)

# Select the columns for the 9 countries and the 10 years
countries = ['China', 'United States', 'India','Russian Federation', 'Japan', 'Germany', 'Indonesia', 'Korea, Rep.']
years = [str(year) for year in range(2010, 2020)]
df_selected = df.loc[df['Country Name'].isin(countries), ['Country Name'] + years]

# Pivot the DataFrame so that the years become the index and the countries become the columns
df_pivot = df_selected.set_index('Country Name').T

# Convert the data type from string to float
df_pivot = df_pivot.astype(float)

# Create an area chart
ax = df_pivot.plot(kind='area', stacked=True, alpha=0.7, figsize=(12, 8))

# Set the chart title and axis labels
ax.set_title('CO2 Emissions(kt)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('% of Total', fontsize=12)

# Set the legend outside the chart area
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# Show the chart
plt.show()


# PLOTTING THE BAR CHART FOR THE TotalCO2 gas emissions (kt) in 2012
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5349736.csv', skiprows=4)

# Select the columns we want
columns = ['Country Name', '2018']
df = df[columns]

# Select the 7 countries of your choice
countries =  ['China', 'United States', 'India','Russian Federation', 'Japan', 'Germany', 'Indonesia', 'Korea, Rep.']
df = df[df['Country Name'].isin(countries)]

# Set the index to the country names
df.set_index('Country Name', inplace=True)

# Plot the bar chart
plt.figure(figsize=(12,6))
plt.bar(df.index, df['2018'])
plt.title('Total CO2 gas emissions(kt) in 2018')
plt.xlabel('Country')
plt.ylabel('Total CO2 gas emissions (kt)')
plt.show()
