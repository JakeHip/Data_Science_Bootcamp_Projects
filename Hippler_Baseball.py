import pandas
import pyodbc
import numpy as np
import matplotlib.pylab as plt

#connect to SQL Server and database
# print(pyodbc.drivers() to determine the Driver
conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server}; '
                      'SERVER= DESKTOP-H83DN8E;'
                      'DATABASE=LahmansBaseballDB;'
                      'UID=jacob;'
                      'Trusted_Connection=yes;')

#run SQL extract to get initial data
cursor = conn.cursor()
cursor.execute('''
SELECT People.playerID , birthYear, birthMonth, BirthDay, BirthCountry, birthState, birthCity, nameFirst, nameLast, nameGiven, weight, height, bats, throws, Batting.*
FROM People
LEFT OUTER JOIN Batting ON People.playerID = Batting.playerID
WHERE finalGame > '2018-03-29'
AND G > 49''')

#for row in cursor:
#    print(row)

#convert data from SQL query to pandas data frame
rows = cursor.fetchall()
rows = [tuple(row) for row in rows]
names = [column[0] for column in cursor.description]
df = pandas.DataFrame(rows, columns= names)

#calculate the age of players by combining year, month, and day columns into a datetime column
# and comparing to the current date. Finally convert the the difference to an int
df['birth_date'] = pandas.to_datetime(dict(year=df.birthYear, month=df.birthMonth, day=df.BirthDay))
df["date_from_birth"] = (pandas.to_datetime('now') - df['birth_date'])
df['age'] = df['date_from_birth'].astype('timedelta64[Y]').astype(int)

#concatenate first and last name columns
df['name'] = df['nameFirst'] + " " + df['nameLast']

#remove columns that are no longer necissary
df.drop(['birth_date', 'date_from_birth', 'nameFirst', 'nameLast', 'birthYear', 'birthMonth', 'BirthDay'], axis=1, inplace=True)

#Which active player had the most runs batted in (“RBI” from the Batting table) from 2015-2018?
#1. create boolean dataframe to narrow down the years between 2015 and 2018 (in thsis data set, that is anything after 2014)
#2. use boolean df to index main data frame
#3. with data frame of only to find the row with the max rbi
year_after_2014 = df['yearID'] > 2014
df_ya2014 = df[year_after_2014]
rbi_max = df_ya2014['RBI'] == df_ya2014['RBI'].max()
df_max = df_ya2014[rbi_max]
print(df_max[['name', 'yearID', 'playerID']])
#Player with highest RBI between 2015 and 2018 = Nolan Arenado, 2016


#How many double plays did Albert Pujols ground into (“GIDP” from Batting table) in 2016?
#use logical operators to find where the name equals 'Albert Pujuols" and the yearID equals 2016
AP_2016 = df[(df['name'] == 'Albert Pujols') & (df['yearID'] == 2016)]
print(AP_2016['GIDP'])
#Albert Pulols's GIDP in 2016 = 24

#Histogram of total '3B'
plt.hist(df['3B'])
plt.title('Histogram of Triples')
plt.show()
plt.clf()

#Histogram of '3B' broken up by year
df.hist(by= 'yearID', column= '3B', bins= 5)
plt.title('Histogram of Triples')
plt.subplots_adjust(left=.05, bottom=.05, right=.95, top=.95, wspace=.4, hspace=.95)
plt.show()
plt.clf()

#Scatter plot of '3B' vs 'SB'
plt.scatter(df['3B'], df['SB'])
plt.xlabel('Steals')
plt.ylabel('Triples')
plt.title('Triples (3B) vs Steals (SB)')
plt.show()
plt.clf()

#Groubed bar charts comparing stats of players over 29 and players 29 and younger
df_2018_over29 = df[(df['yearID'] == 2018) & (df['age'] > 29)]
df_2018_under30 = df[(df['yearID'] == 2018) & (df['age'] < 30)]
labels = ['R', 'H', 'RBI', 'HR']
over_29= [df_2018_over29['R'].mean(), df_2018_over29['H'].mean(), df_2018_over29['RBI'].mean(), df_2018_over29['HR'].mean()]
under_30 = [df_2018_under30['R'].mean(), df_2018_under30['H'].mean(), df_2018_under30['RBI'].mean(), df_2018_under30['HR'].mean()]
x = np.arange(len(labels))
fig, ax = plt.subplots()
rects1 = ax.bar(x - .35/2, under_30, .35, label = 'Under Age 30')
rects2 = ax.bar(x + .35/2, over_29, .35, label = 'Age 30 or Over')
ax.set_ylabel('Stats')
ax.set_title('Average Stats by Age')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

#scatter plot of home runs vs intentional walks
plt.scatter(df['HR'], df['IBB'])
plt.xlabel('Home Runs')
plt.ylabel('Intentional Base on Balls (Intentional Walk)')
plt.title('Home Runs vs Walks')
plt.show()
plt.clf()


#Based on the earlier search, the players with the top 3 RBIs are Nolan Arenado, J.D. Martinez, and Giancarlo Stanton.
#this plot shows how their RBIs have flucuated throughout the years.
df_NA = df[(df['name'] == 'Nolan Arenado')]
df_JDM = df[(df['name'] == 'J. D. Martinez')]
df_GS = df[(df['name'] == 'Giancarlo Stanton')]
plt.plot(df_NA['yearID'], df_NA['RBI'], label = "Nolan Arenado")
plt.plot(df_JDM['yearID'], df_JDM['RBI'], label = "J.D. Martinez")
plt.plot(df_GS['yearID'], df_GS['RBI'], label = "Giancarlo Stanton")
plt.xlabel('Year')
plt.ylabel('RBI')
plt.title('RBI over the Years')
plt.legend()
plt.show()