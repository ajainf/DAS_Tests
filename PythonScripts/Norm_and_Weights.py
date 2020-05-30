'''
author: AJF
date: May 26, 2020
The purpose of this code is to create several plots and calculations to test residuals

'''

import scipy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm



def  resid_boxplot(varname, norm_residual, valmin, valmax, fits, savepath):
    #------------------------------------- Box plot
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.boxplot(norm_residual)
    ax1.plot(1, np.nanmean(norm_residual), 'x')

    plt.ylabel('Normalized Residual Values')
    plt.title(varname + " Residuals Box Plot")
    plt.ylim([valmin,valmax])

    plt.savefig(savepath+'resid_boxplot.png', format='png', dpi=300)
    #print('saved boxplot')
    
    


def  resid_plot(varname, norm_residual, valmin, valmax, fits, savepath):
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot(1, 1, 1)
    #------------------------------------- Residual

    ax2.scatter(fits, norm_residual, color='k', marker='o', s=4, label='residuals') # s= size
    ax2.plot(( min(fits),max(fits) ), (0, 0), 'r-', label='zero')
    ax2.plot(( min(fits),max(fits) ), (np.nanmean(norm_residual), np.nanmean(norm_residual)), 'b-', label='mean')
    
    plt.xlabel('Estimated Values')
    plt.ylabel('Normalized Residual')
    plt.title( "".join([varname, ' Residual Plot' ]))
    plt.ylim([valmin,valmax])
    plt.legend()

    plt.savefig(savepath+'resid_plot.png', format='png', dpi=300)
    #print('saved residual plot')



def  resid_pdf(varname, norm_residual, valmin, valmax, fits, savepath):
    fig = plt.figure(figsize=(5,5))
    ax3 = fig.add_subplot(1, 1, 1)
    #------------------------------------- PDF

    # Fitting a distribution
    _, bins, _ = ax3.hist(norm_residual, 21, density=200, alpha=0.7)
    mu, sigma = scipy.stats.norm.fit(norm_residual)
    best_fit = scipy.stats.norm.pdf(bins, mu, sigma)
    #print('Fitted Normal Dist. mean=%.3f stdev=%.3f' % (mu, sigma))

    # Standard Gaussian
    mean = 0; std = 1; variance = np.square(std)
    x = np.arange(valmin,valmax,.01)
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    # Plot
    ax3.plot(bins, best_fit, label='Fitted Normal Distribution')
   #ax3.plot(x,f, color='green', label='Standard Gaussian Distribution')
    ax3.legend()
    plt.title('Fitting a Normal Distribution to ' +varname+' Residuals')

    plt.savefig(savepath+'resid_pdf.png', format='png', dpi=300)
    #print('saved pdf plot')


                    
def  resid_histogram(varname, norm_residual, valmin, valmax, fits, savepath):
    fig = plt.figure(figsize=(5,5))
    ax4 = fig.add_subplot(1, 1, 1)
    #------------------------------------- Histogram
    _, bins, _ = ax4.hist(norm_residual, color = 'blue', edgecolor = 'black', alpha = 0.5, bins = 21)

    plt.xlim([valmin,valmax])
    plt.title(varname + ' Residuals Histogram')
    plt.ylabel('Frequency')
    plt.xlabel('Normalized Residual')

    plt.savefig(savepath+'resid_histogram.png', format='png', dpi=300)
    #print('saved histogram')



def  resid_cdf(varname, norm_residual, valmin, valmax, fits, savepath):
    fig = plt.figure(figsize=(5,5))
    ax= fig.add_subplot(1, 1, 1)
    #-------------------------------------CDF
    data = norm_residual
    length = len(data)
    
    ax.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
    mean = 0; std = 1; variance = np.square(std)
    #ax.plot(np.sort(scipy.stats.norm.rvs(loc=mean, scale=variance, size=length)), np.linspace(0, 1, len(data), endpoint=False))
    
    plt.ylabel('Probability')
    plt.legend('top right')
    plt.legend(['Data'])
    #plt.legend(['Data', 'Theoretical Values'])
    plt.title(varname+' CDF')

    plt.savefig(savepath+'resid_cdf.png', format='png', dpi=300)
    #print('saved cdf plot')
                    


def  resid_qq(varname, norm_residual, valmin, valmax, fits, savepath):
    fig = plt.figure(figsize=(5,5))
    ax6= fig.add_subplot(1, 1, 1)
    #------------------------------------- QQplot
    sm.qqplot(norm_residual, line='s', ax=ax6)
    plt.title("Normal Q-Q plot for Residuals")
    plt.ylim([valmin,valmax])

    plt.savefig(savepath+'resid_qq.png', format='png', dpi=300)
    #print('saved qq plot')


# -------------------------------------Statistical Normality Test
def stats_norm_test(varname, norm_residual):

    from scipy.stats import normaltest
    import numpy as np
    import pandas as pd
    
    print(varname)
    
    # -- Summarize
    meanval = np.nanmean(norm_residual)
    stdval = np.nanstd(norm_residual)
    
    print('mean=%.3f' % np.nanmean(norm_residual))
    print('stdv=%.3f' % np.nanstd(norm_residual))


    # -- Normality test
    stat, p = normaltest(norm_residual)

    print()
    print('Normality Test from scipy.stats')
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    # -- Interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (p>'+str(alpha)+')') # (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (p<'+str(alpha)+')') # reject H0
        
    return meanval, stdval




# -------------------------------------Residual Check

def residual_checks(varname,res,dat,mod, savepath):
    import pandas as pd
    import numpy as np
    
    #mod = mod[mod>0].dropna() # positivity checks; drop 0
    df  = pd.concat([res, dat, mod], axis=1)
    #df = df.dropna() # carefull with the drops
    df.columns =['res','dat','mod']

    # Create flat arrays
    norm_residual = np.array(df.loc(axis=1)['res']) #misfit 
    data = np.array(df.loc(axis=1)['dat'])#dat
    fits = np.array(df.loc(axis=1)['mod'])#mod
    
    # removing nans -- but creates mismatched shapes
    #norm_residual = norm_residual[~np.isnan(norm_residual)]
    #data = data[~np.isnan(data)]
    #fits = fits[~np.isnan(fits)]
    
    # make nans zero
    norm_residual[np.isnan(norm_residual)]=0
    data[np.isnan(data)]=0
    fits[np.isnan(fits)]=0
    

    # Residuals
    print('Residual Stats')
    # Stats normality Check
    resid_mean, resid_stdev = stats_norm_test(varname, norm_residual)

    # Plots for Residuals
    resid_plot(varname, norm_residual, min(norm_residual), max(norm_residual), fits, savepath)
    resid_cdf(varname, norm_residual, min(norm_residual), max(norm_residual), fits, savepath)
    resid_pdf(varname, norm_residual, min(norm_residual), max(norm_residual), fits, savepath)
    resid_qq(varname, norm_residual, min(norm_residual), max(norm_residual), fits, savepath)

    # add to excel file
    

    return resid_mean, resid_stdev, norm_residual, data, fits
