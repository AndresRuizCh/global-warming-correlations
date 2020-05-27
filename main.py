import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import geopandas as gpd
from datetime import datetime
import scipy
import matplotlib
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
#
# Matplotlib Settings
rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
#
# External files routes
shapefile = 'data\\shapes.shp'  # Obtained from https://github.com/nvkelso/natural-earth-vector
worldbank = 'data\\WorldBank.csv'  # Obtained from https://databank.worldbank.org/source/world-development-indicators#
tempdata = 'data\\Temperatures.csv'  # Obtained from https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
countrycodes = 'data\\CountryCodes.xlsx'  # Obtained from https://datahub.io/core/country-list
#
# Function to merge all the data into one only dataframe
def mergedata(shapefile_column='ADM0_A3'):
    ''' Merge all the datasets into one.
        - shapefile_column : (str) Name of the name column of the .shp file '''
    #
    # Use geopandas to read the shapefiles for the maps
    gdf = gpd.read_file(shapefile)[[shapefile_column, 'geometry']]
    gdf = gdf.rename({shapefile_column:'Country Code'}, axis=1)
    #
    # Read the World Bank data and melt into one large dataframe
    data = pd.read_csv(worldbank)
    data.columns = ['Country Name', 'Country Code', 'Series Name', 'Series Code'] + list(np.arange(1960, 2020))
    data = data.melt(id_vars = data.columns[:4], value_vars = data.columns[4:], var_name='Year', value_name='Value')
    #
    # Read the temperatures dataset and clean the country codes and rename the columns adding the Series columns
    temperatures = pd.read_csv(tempdata)
    codes = pd.read_excel(countrycodes)
    temperatures['Country Code'] = temperatures[['Country']].merge(codes, how='left', on='Country')['Country Code']
    temperatures = temperatures.drop('AverageTemperatureUncertainty', axis=1).dropna()
    temperatures['Year'] = pd.to_datetime(temperatures['dt']).apply(lambda x: x.year)
    temperatures = temperatures.groupby(['Country', 'Country Code', 'Year']).mean().reset_index()
    temperatures = temperatures.rename({'AverageTemperature': 'Value'}, axis=1)
    temperatures['Series Name'] = 'Average Temperature'
    temperatures['Series Code'] = 'AVG.TMP'
    temperatures = temperatures.drop('Country', axis=1)
    #
    # Merge the three dataframes into the same dataframe
    aux = data[['Country Code', 'Country Name']].drop_duplicates()
    aux = temperatures[['Country Code']].merge(aux, on='Country Code', how='left')
    temperatures['Country Name'] = aux['Country Name']
    data = pd.concat([data, temperatures], sort=True)
    return data, gdf

def plot(data, code, name, function, maxyears=50):
    ''' Plot the time evolution of some Series.
        - data : (pd.DataFrame) Data from mergedata function
        - code : (str) Series code
        - name : (str) series name1
        - function : (function) Python function to use if necesary (e.g. averaging the data)
        - maxyears : (int) Future years to plot using Linear Regression'''
    #
    # Clean the data to filter the desired code
    df = data[(data['Series Code'] == code) & (data['Year']>1900)].groupby('Year').agg(function)
    df = df.rename({'Value':name}, axis=1)
    X = df[name].dropna().reset_index().values
    x = X[:,0].reshape((-1, 1))
    y = X[:,1]
    #
    # Perform a Linear Regression analysis
    model = LinearRegression()
    model.fit(x, y)
    new_x = np.arange(X[0,0], X[-1,0]+maxyears).reshape((-1, 1))
    new_y = model.predict(new_x)
    #
    # Plot data and save figures
    fig, ax = plt.subplots(figsize=(4,3), ncols=1)
    ax.scatter(x, y)
    ax.plot(new_x, new_y, color='r')
    ax.set_xlabel('Year')
    ax.set_ylabel(name)
    plt.tight_layout()
    plt.savefig('Figures\\Plot_' + name + '.pdf')

def plotmap(data, gdf, code, name, function, exceptions=None):
    ''' Plot a choropleth map of some Series.
        - data : (pd.DataFrame) Data from mergedata function
        - gdf : (geopandas.GeoDataFrame) GeoDataFrame object from mergedata function
        - code : (str) Series code
        - name : (str) series name1
        - function : (function) Python function to use if necesary (e.g. averaging the data)
        - exceptions : (list) List of countries to exclude (e.g Greenland)'''
    #
    # Prepare the dataframe and sort values
    df = data[(data['Series Code'] == code) & (data['Year']>1900)]
    if exceptions:
        df = df[~df['Country Code'].isin(exceptions)]
    df = df.sort_values(['Country Code', 'Year']).groupby('Country Code').agg(function).reset_index()
    df = gdf.merge(df, on='Country Code', how='left')
    #
    # Plot data and save figures
    ax = df.dropna().plot(column='Value', cmap='winter', figsize=(8,4), legend=True)
    ax.set_axis_off()
    ax.get_figure()
    plt.tight_layout()
    plt.savefig('Figures\\Map_' + name + '.pdf')

def correlations(data, code1, code2, name1, name2, function=None, full=False):
    ''' Calculate the correlations between two series.
        - data : (pd.DataFrame) Data from mergedata function
        - code1 : (str) Series Code 1
        - code2 : (str) Series Code 2
        - name1 : (str) Series Name 1
        - name2 : (str) Series Name 2
        - function : (function) Python function to use if necesary (e.g. averaging the data)
        - full : (bool) Perform or not a detailed correlation with all the possible data'''
    #
    # Clean and filter the data depending on the detail of it
    if full:
        a = data[(data['Series Code'] == code1)].rename({'Value':name1}, axis=1)[['Country Code', 'Year', name1]]
        b = data[(data['Series Code'] == code2)].rename({'Value':name2}, axis=1)[['Country Code', 'Year', name2]]
        a = a.set_index(['Country Code', 'Year'])
        b = b.set_index(['Country Code', 'Year'])
        df = a.join(b).dropna().reset_index()
    else:
        a = data[(data['Series Code'] == code1)].groupby('Year').agg(function).rename({'Value':name1}, axis=1)
        b = data[(data['Series Code'] == code2)].groupby('Year').agg(function).rename({'Value':name2}, axis=1)
        df = pd.concat([a,b], sort=True, axis=1).dropna().reset_index()
    #
    # Plot data
    fig, ax = plt.subplots(figsize=(4,3), ncols=1)
    ax.scatter(df[name1], df[name2])
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    #
    # Calculate the correlations
    z = np.polyfit(df[name1], df[name2], 1)
    p = np.poly1d(z)
    corr = np.corrcoef(x=df[name1], y=df[name2])[0,1]
    ax.plot(df[name1],p(df[name1]), color='red', ls='--', alpha=0.5)
    ax.set_ylim(df[name2].min(), df[name2].max())
    ax.set_title('corr=' + str(np.round(corr, 4)))
    #
    # Save Figures
    plt.tight_layout()
    plt.savefig('Figures\\Corr_' + name1 + '_' + name2 + '.pdf')

if __name__=='__main__':
    # Merge data
    data, gdf = mergedata()
    # Average Temperataure
    plot(data, 'AVG.TMP', 'Average Temperature', 'mean')
    plotmap(data, gdf, 'AVG.TMP', 'Temperature Increase', lambda x: x.nlargest(20).mean() - x.nsmallest(20).mean(), ['DNK'])
    # Natural Disasters
    plotmap(data, gdf, 'EN.CLC.MDAT.ZS', 'Disasters (Percent of population affected)', 'max')
    correlations(data, 'AVG.TMP', 'EN.CLC.MDAT.ZS', 'Average Temperature', 'Disasters (Population affected)', full=True)
    # Carbon Dioxide
    plotmap(data, gdf, 'EN.ATM.CO2E.KT', 'CO2 Emitted (kT)', 'mean')
    plot(data, 'EN.ATM.CO2E.KT', 'CO2 Emitted (kT)', 'mean')
    correlations(data, 'AVG.TMP', 'EN.ATM.CO2E.KT', 'Average Temperature (Celsius)', 'CO2 Emitted (kT)', 'mean')
    # Greenhouse Gases
    plotmap(data, gdf, 'EN.ATM.GHGO.KT.CE', 'Greenhouse Gases Emitted (kT)', 'mean')
    plot(data, 'EN.ATM.GHGO.KT.CE', 'Greenhouse Gases Emitted (kT)', 'mean')
    correlations(data, 'AVG.TMP', 'EN.ATM.GHGO.KT.CE', 'Average Temperature', 'Greenhouse Gases Emitted (kT)', 'mean')
    # Nytrogen OXides
    plotmap(data, gdf, 'EN.ATM.NOXE.KT.CE', 'Nitrogen Oxides', 'mean')
    plot(data, 'EN.ATM.NOXE.KT.CE', 'Nitrogen Oxides', 'mean')
    correlations(data, 'AVG.TMP', 'EN.ATM.NOXE.KT.CE', 'Average Temperature', 'Nitrogen Oxides', 'mean')
    # Population Growth
    plotmap(data, gdf, 'SP.POP.GROW', 'Population Growth', 'mean')
    plot(data, 'SP.POP.GROW', 'Population Growth', 'mean')
    correlations(data, 'AVG.TMP', 'SP.POP.GROW', 'Average Temperature', 'Population Growth', full=True)
    # Poverty
    plotmap(data, gdf, 'SI.POV.DDAY', 'Poverty Ratio', 'mean')
    plot(data, 'SI.POV.DDAY', 'Poverty Ratio', 'mean')
    correlations(data, 'AVG.TMP', 'SI.POV.DDAY', 'Average Temperature', 'Poverty Ratio', full=True)
