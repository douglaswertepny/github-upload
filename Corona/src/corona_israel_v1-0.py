'''
Created on Feb 27, 2020
Updated on Jun 18, 2020

Corona_israel_v1.1:
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
rolling_mean = 7 # Number of days used for the rolling mean of new cases
name ='Corona Data - wiki.csv' # Imported file name

######################## Imports Packages #############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################## Loads in and Manipulates initial Data #############################
# Loading data usecols='Date'
df = pd.read_csv(name,parse_dates=['Date'],delimiter=';')
# Creates new data sets from imported data, 'Active Cases' and 'Death Rate'
df1 = pd.DataFrame(df['Total Cases'] -df['Deaths'] -df['Recoveries'], columns=['Active Cases'])
df2 = pd.DataFrame(df['Deaths']/(df['Deaths']+df['Recoveries']), columns=['Death Rate'])
df = pd.concat([df, df1,df2], axis=1, sort=False)

######################## Calculates new cases ##########################
df3 = [df['Total Cases'][0]]
i=1
while i < df.shape[0]:
    dft = [df['Total Cases'][i]-df['Total Cases'][i-1]]
    df3 = np.vstack((df3,dft))
    i+=1
df3 = pd.DataFrame(df3,columns=['New Cases'])
df = pd.concat([df,df3],axis=1,sort=False)

######################## Calculates net change in cases ##########################
df4 = [df['Active Cases'][0]]
i=1
while i < df.shape[0]:
    dft = [df['Active Cases'][i]-df['Active Cases'][i-1]]
    df4 = np.vstack((df4,dft))
    i+=1
df4 = pd.DataFrame(df4,columns=['Net Change Cases'])
df = pd.concat([df,df4],axis=1,sort=False)

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
        elif x == 'Active Cases' or x== 'Death Rates' or x== 'New Cases':
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
    return df_fit

# Plots the data
def ploting_data(df,log):
    data_y = ['Total Cases','Active Cases','Recoveries','Deaths']
    data_color = ['bo','go','yo','ro']
    i=0
    while i <= 3:
        plt.plot(df['Date'],df[data_y[i]],data_color[i],label=data_y[i])
        i += 1  
    plt.ylabel('Number of Cases')
    plt.gcf().autofmt_xdate()
    plt.legend()
    if log == True:
        plt.yscale('log')
        plt.title('Corona in Israel - Log Scale')
        plt.savefig('Corona in Israel - Log Scale.png')
    else:
        plt.title('Corona in Israel - Linear Scale')
        plt.savefig('Corona in Israel - Linear Scale.png')
    plt.show()
    return

# Plots the data and the fit
def ploting_data(df,df_fit,log,fit):
    data_y = ['Total Cases','Active Cases','Recoveries','Deaths']
    data_color = ['b','g','y','r']
    i=0
    if fit == True:
        while i <= 3:
            plt.plot(df['Date'],df[data_y[i]],data_color[i] + 'o',label=data_y[i])
            plt.plot(df_fit['Date'],df_fit[data_y[i]],data_color[i])
            i += 1  
        plt.ylabel('Number of Cases')
        plt.gcf().autofmt_xdate()
        plt.legend()
        if log == True:
            plt.yscale('log')
            plt.title('Corona in Israel - Log Scale - Fit')
            plt.savefig('Corona in Israel - Log Scale - Fit.png')
        else:
            plt.title('Corona in Israel - Linear Scale - Fit')
            plt.savefig('Corona in Israel - Linear Scale - Fit.png')
    else:
        while i <= 3:
            plt.plot(df['Date'],df[data_y[i]],data_color[i] + 'o',label=data_y[i])
            i += 1  
        plt.ylabel('Number of Cases')
        plt.gcf().autofmt_xdate()
        plt.legend()
        if log == True:
            plt.yscale('log')
            plt.title('Corona in Israel - Log Scale')
            plt.savefig('Corona in Israel - Log Scale.png')
        else:
            plt.title('Corona in Israel - Linear Scale')
            plt.savefig('Corona in Israel - Linear Scale.png')
    plt.show()
    return

######################## Calculates Results Using Functions #############################
# Fits the data
df_fit = exponential_fit(df,day_fits,day_pro)
# Prints out the rate of growth results
print('Most Recenent Data: ' + str(df['Date'][len(df['Date'])-1]))
for x in ['Total Cases','Recoveries','Deaths']:
    print(rate_of_growth(df,x,day_fits)[2])
print('As of ' + str(df['Date'][df.index[-1]]) + ':')
print('\tTotal Cases:\t' + str(df['Total Cases'][df.index[-1]]))
print('\tRecoveries\t' + str(df['Recoveries'][df.index[-1]]))
print('\tActive Cases:\t' + str(df['Active Cases'][df.index[-1]]))
print('\tNew Cases:\t' + str(df['New Cases'][df.index[-1]]))
print('\tDeaths:\t\t' + str(df['Deaths'][df.index[-1]]))
print('\tDeath Rate:\t' + str(round(100*df['Death Rate'][df.index[-1]],1))+'%')
print('\tIncrease of active cases:\t' + str(round(100*(df['Active Cases'][df.index[-1]]-df['Active Cases'][df.index[-2]])/df['Active Cases'][df.index[-2]]))+'%')
######################## Plots Figures #############################
# Plotting Raw Data
ploting_data(df,df_fit,True,False)
ploting_data(df,df_fit,False,False)
ploting_data(df,df_fit,True,True)
ploting_data(df,df_fit,False,True)

# Plotting Death Rate
plt.plot(df['Date'],100*df['Death Rate'],'bo',label='Death Rate')
plt.title('Corona in Israel - Precent Death Rate')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - Death Rate.png')
plt.show()
# Plotting Death Rate - Fit
#plt.plot(df['Date'],100*df['Death Rate'],'bo',label='Death Rate - Fit')
#plt.plot(df_fit['Date'],100*df_fit['Death Rate'],'b')
#plt.title('Corona in Israel - Precent Death Rate - Fit')
#plt.gcf().autofmt_xdate()
#plt.savefig('Corona in Israel - Death Rate - Fit.png')
#plt.show()
# Plotting New Cases
plt.plot(df['Date'],df['New Cases'],'bo',label='New Cases')
plt.title('Corona in Israel - New Cases')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - New Cases.png')
plt.show()

# Plotting New Cases
plt.plot(df['Date'],df['New Cases'].rolling(rolling_mean).mean(),'bo',label='New Cases')
plt.title('Corona in Israel - New Cases - ' + str(rolling_mean) + ' Day Rolling Mean')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - New Cases - Rolling Average.png')
plt.show()

# Plotting Net Change in Cases
plt.plot(df['Date'],df['Net Change Cases'],'bo',label='Net Change in Cases')
plt.title('Corona in Israel - Net Change in Cases')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - Net Change in Cases.png')
plt.show()

# Plotting Net Change in Cases
plt.plot(df['Date'],df['Net Change Cases'].rolling(rolling_mean).mean(),'bo',label='Net Change in Cases')
plt.title('Corona in Israel - Net Change in Cases - ' + str(rolling_mean) + ' Day Rolling Mean')
plt.gcf().autofmt_xdate()
plt.savefig('Corona in Israel - Net Change in Cases - Rolling Average.png')
plt.show()