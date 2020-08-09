from sklearn.datasets import load_boston
import pandas as pd
import numpy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import tt_ind_solve_power
import matplotlib.pylab as plt

#need to import the data set which can be done using load_boston()
#load_boston() only loads 13 columns but doesn not load the MEDV column
#the MEDV column can be added by looking for the boston.target column
#the MEDV column can then be concatenated onto the load_boston data set
boston = load_boston()
boston_df_prep = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df_medv = pd.DataFrame(boston.target, columns=['MEDV'])
boston_df = pd.concat([boston_df_prep, boston_df_medv], axis = 1)
list(boston_df.columns)
print(boston_df.head())
print(boston_df)

#1.1 Compute the mean and standard deviation of the variable
print("Mean per capita crime rate")
print(boston_df['CRIM'].mean())
print("Standard deviation of per capita crime rate")
print(boston_df['CRIM'].std())
#Pandas dataframe also allow for the means and standard deviations of all columns to be done in one function
#print(boston_df.mean())
#print(boston_df.std())

#1.2 Plot a histogram of the variable.
plt.hist(boston_df['CRIM'])
plt.title('Histogram of Per Capita Crime Rate')
plt.show()
plt.clf()

#1.3 What is the sample correlation between your chosen variable and median home price?
print("Correlation of Per Capita Crime Rate and Median Value")
print(boston_df['CRIM'].corr(boston_df['MEDV']))

#1.4 Perform a regression, predicting MEDV from your chosen variable?
reg = sm.OLS(boston_df['CRIM'], boston_df['MEDV']).fit()
print(reg.summary())



#2.1 What is the null hypothesis?
#Null Hypothesis (H0): The tracts that border the Charles River (CHAS) will
#   not have a different median price (MEDV) than those that do.
#Alternative Hypothesis (H1): The tracts that border the Charles River (CHAS) 
#  will have a higher median price (MEDV) than those that do not

#function to calculate statistical values such as p-value, and CI
def t_one_sample(samp, mu = 0.0, alpha = 0.05):
    t_stat = stats.ttest_1samp(samp, mu)
    scale = numpy.std(samp)
    loc = numpy.mean(samp)
    ci = stats.t.cdf(alpha/2, len(samp), loc=mu, scale=scale)
    print('Results of one-sample two-sided t test')
    print('Mean         = %4.3f' % loc)
    print('t-Statistic  = %4.3f' % t_stat[0])
    print('p-value      < %4.3e' % t_stat[1])
    print('On degrees of freedom = %4d' % (len(samp) - 1))
    print('Confidence Intervals for alpha =' + str(alpha))
    print('Lower =  %4.3f Upper = %4.3f' % (loc - ci, loc + ci))
    
boston_df_chas = boston_df[boston_df['CHAS'] == 1.0]
#print(boston_df_chas)
t_one_sample(boston_df_chas['MEDV'])

#2.2 Calculate the p-value. Use the sample mean of the target as an estimate of the population mean.
# 6.874e-16- much smaller than .05, so we would reject the null hypothesis

#2.3 What is the 90% confidence interval for the target (price) of tracts that border the Charles River?
#Confidence Intervals for alpha =0.05
#Lower =  27.939 Upper = 28.941

#2.4 Assume an effect size (Cohen’s d) of 0.6. If you want 80% power, what group size is necessary?
power = tt_ind_solve_power(effect_size=0.6, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
print("Size necessary for 80% power with .6 effect size: " + str(power))
#44.585


#using multiline sting as comment for 3rd question
''' 
Imagine you are the city planner of Boston and can add various new features 
to each census tract, such as a park. Be creative with your new “features”
 – we use the term loosely. You can assume that none of the tracts contained 
 your features previously. Design an experiment to explore the effects of
these features on the media house price in census tracts. You should include 
an explanation of the experimental design as well as a plan of analysis, which 
should include a discussion of group size and power. Be sure to apply the 
knowledge you learned in the Data Science Research Methods courses.'

I would want to test the effect having a hospital 'nearby' has on the median 
home price. To test this you would first have to determine what nearby is 
considered. In this case we will define nearby as within 10 miles. We could 
then use a binary response (similar to CHAS) to determine if the home is within
10 miles of a hospital and run a regression test to determine if that may have
an effect of the meadian house price.
 Since you wouldn't necessarily know what the effect size is, 
you would have to determine what size of effect you want to be able to see in 
order to determine the size of the population you need to sample. In this case,
lets assume you want to find even small effects, which mean the samllest d value
you would be looking for would be .2. You could then use the tt_ind_solve_power
function using .2 as the effect size, and .8 as the power to determine what 
the sample size would need to be. In this case you would need a sample of 394.
power = tt_ind_solve_power(effect_size=0.2, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
print("Size necessary for 80% power with .2 effect size: " + str(power))
'''
