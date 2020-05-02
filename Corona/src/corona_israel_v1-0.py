'''
Created on Feb 27, 2020

Corona_israel_v1.0:
Plots the following information about Corona: Total Cases, Recoveries, Deaths, and Active Cases.
Extrapolates assuming local exponential growth for Total Cases, Recoveries, and Deaths.
Death Rate is calculate by only considering Deaths and Recoveries.

Inputs:
    number of days used to fit the data: day_fits
    number of days the program projects out to: day_pro
    file name: name

Outputs:
Prints out results for: Total Cases, Recoveries, Deaths
    Daily Precent Increase
    Doubling Time
Prints and saves graphs:
    Total Cases, Active Cases, Deaths, Recoveries - Log, Linear - Fits, no fits
    Death Rate - Fit, no fit

@author: Douglas Wertepny
'''
######################## User Parameters #############################
day_fits = 5 # Number of days the program uses to fit the data
day_pro = 5 # Number of days the fits project
name ='Corona Data.csv' # Imported file name

######################## Imports Packages #############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################## Loads in and Manipulates initial Data #############################
# Loading data usecols='Date'
df = pd.read_csv(name,parse_dates=['Date'])
# Creates new data sets from imported data, 'Active Cases' and 'Death Rate'
df1 = pd.DataFrame(df['Total Cases'] -df['Deaths'] -df['Recoveries'], columns=['Active Cases'])
df2 = pd.DataFrame(df['Deaths']/(df['Deaths']+df['Recoveries']), columns=['Death Rate'])
df = pd.concat([df, df1,df2], axis=1, sort=False)

######################## Defines Functions #############################
# Calculating rate of growth 
def rate_of_growth(df,name,n):
    'Calculates the rate of growth of a column of datagram, df[name], using the last n data points'
    x_axis = list(range(n))
    df_n = np.array(df[name].tail(n))
    df_log = np.log(df_n)
    linfit, errorfit =np.polyfit(x_axis,df_log,1,cov=True)
    rate = round(100*linfit[0],1)
    error = round(100*np.sqrt(errorfit[0,0]),1)
    doubling_time = round(np.log(2)/linfit[0],1)
    #datay_fit0 = np.exp(linfit[0]*df_log + linfit[1])
    string_0 = name + ' information from last ' + str(n) + ' days:\n'
    string_1 = '\tDaily Precent Increase: ' + str(rate) + u"\u00B1" + str(error) + '%'
    string_2 = '\n\tDoubling Time: ' + str(doubling_time) + ' days'
    return linfit, errorfit, string_0+string_1+string_2
# Calculating the exponential fit
def exponential_fit(df,n,m):
    'Fits and exponential curve to the data in df using the last n points and extrapolating for m points. Calcuates Active Cases by subtracting Deaths and Recoveries from Total Cases'
    df = df.tail(n)
    df = df.reset_index(drop=True)
    x_axis = range(n)
    x_fit_axis = range(n+m)
    for x in list(df):
        if x == 'Date':
            date_fit = pd.date_range(start=df['Date'][0],periods=n+m)
            df_temp = pd.DataFrame(date_fit.to_series(),columns=['Date'])
            df_fit = df_temp.reset_index(drop=True)
        elif x == 'Active Cases' or x== 'Death Rates':
            pass
        else:
            y_log = np.log(np.array(df[x]))
            linfit, errorfit =np.polyfit(x_axis,y_log,1,cov=True)
            y_fit = linfit[0]*x_fit_axis + linfit[1]
            y_fit = np.exp(y_fit)
            df_temp = pd.DataFrame(y_fit,columns=[x])
            df_fit = pd.concat([df_fit,df_temp], axis=1, sort=False)
    df_temp1 = pd.DataFrame(df_fit['Total Cases']-df_fit['Deaths']-df_fit['Recoveries'],columns=['Active Cases'])
    df_temp2 = pd.DataFrame(df_fit['Deaths']/(df_fit['Deaths']+df_fit['Recoveries']),columns=['Death Rates'])
    df_fit = pd.concat([df_fit,df_temp1], axis=1, sort=False)
    df_fit = pd.concat([df_fit,df_temp2], axis=1, sort=False)
    print(df_temp2)
    return df_fit

######################## Calculates Results Using Functions #############################
# Fits the data
df_fit = exponential_fit(df,day_fits,day_pro)
# Prints out the rate of growth results
print('Most Recenent Data: ' + str(df['Date'][len(df['Date'])-1]))
for x in ['Total Cases','Recoveries','Deaths']:
    print(rate_of_growth(df,x,day_fits)[2])
    
######################## Plots Figures #############################
# Plotting Raw Data Log
plt.plot(df['Date'],df['Total Cases'],'bo',label='Total Cases')
plt.plot(df['Date'],df['Active Cases'],'go',label='Active Cases')
plt.plot(df['Date'],df['Recoveries'],'yo',label='Recoveries')
plt.plot(df['Date'],df['Deaths'],'ro',label='Deaths')
plt.title('Corona in Israel - Logarithmic Scale')
plt.ylabel('Number of Cases')
plt.yscale('log')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('Corona in Israel - Logarithmic Scale.png')
plt.show()
# Plotting Fit Data Log - Fit
plt.plot(df['Date'],df['Total Cases'],'bo',label='Total Cases')
plt.plot(df_fit['Date'],df_fit['Total Cases'],'b') #,label='Total Cases - Fit'
plt.plot(df['Date'],df['Active Cases'],'go',label='Active Cases')
plt.plot(df_fit['Date'],df_fit['Active Cases'],'g')
plt.plot(df['Date'],df['Recoveries'],'yo',label='Recoveries')
plt.plot(df_fit['Date'],df_fit['Recoveries'],'y')
plt.plot(df['Date'],df['Deaths'],'ro',label='Deaths')
plt.plot(df_fit['Date'],df_fit['Deaths'],'r')
plt.title('Corona in Israel - Logarithmic Scale - Fits')
plt.ylabel('Number of Cases')
plt.yscale('log')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('Corona in Israel - Logarithmic Scale - Fit.png')
plt.show()

# Plotting Raw Data Linear
plt.plot(df['Date'],df['Total Cases'],'bo',label='Total Cases')
plt.plot(df['Date'],df['Active Cases'],'go',label='Active Cases')
plt.plot(df['Date'],df['Recoveries'],'yo',label='Recoveries')
plt.plot(df['Date'],df['Deaths'],'ro',label='Deaths')
plt.title('Corona in Israel - Linear Scale')
plt.ylabel('Number of Cases')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('Corona in Israel - Linear Scale.png')
plt.show()
# Plotting Raw Data Linear - Fit
plt.plot(df['Date'],df['Total Cases'],'bo',label='Total Cases')
plt.plot(df_fit['Date'],df_fit['Total Cases'],'b')
plt.plot(df['Date'],df['Active Cases'],'go',label='Active Cases')
plt.plot(df_fit['Date'],df_fit['Active Cases'],'g')
plt.plot(df['Date'],df['Recoveries'],'yo',label='Recoveries')
plt.plot(df_fit['Date'],df_fit['Recoveries'],'y')
plt.plot(df['Date'],df['Deaths'],'ro',label='Deaths')
plt.plot(df_fit['Date'],df_fit['Deaths'],'r')
plt.title('Corona in Israel - Linear Scale - Fits')
plt.ylabel('Number of Cases')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig('Corona in Israel - Linear Scale - Fit.png')
plt.show()

# Plotting Death Rate
plt.plot(df['Date'],100*df['Death Rate'],'bo',label='Death Rate')
plt.title('Corona in Israel - Precent Death Rate')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - Death Rate.png')
plt.show()
# Plotting Death Rate - Fit
plt.plot(df['Date'],100*df['Death Rate'],'bo',label='Death Rate - Fit')
plt.plot(df_fit['Date'],100*df_fit['Death Rate'],'b')
plt.title('Corona in Israel - Precent Death Rate - Fit')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - Death Rate - Fit.png')
plt.show()