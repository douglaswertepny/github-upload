# github-upload
'''
This is a simple project that reads in an input csv file that contains information on coron in Israel.
The information the file needs to contain is: 'Date', 'Total Cases', 'Recoveries', 'Deaths'
From this initial data it fits exponential growth for a choosen number of days and projects the estimates.
It also calulates the 'Active Cases' and 'Death Rate'
Note that 'Death Rate' is calculated from the NUMBER OF CLOSED CASES not the total number of cases.
'''
