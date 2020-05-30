#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:57:19 2017
Modified on Th Feb 28 2020
@author: Anjuli

This code contains a variety of functions that 
are useful in the ipython notebook workflow
"""
from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy.stats as st



#------------------ MIRCA crop codes

# Crop Codes
MIRCA2000_cropclasses={1: 'wheat', 2: 'maize', 3: 'rice', 4: 'barley', 5: 'rye', 6: 'millet', 7: 'sorghum', 8: 'soybeans', 9: 'sunflower', 10: 'potatoes', 11: 'cassava', 12: 'sugarcane', 13: 'sugar beet', 14: 'oil palm', 15: 'rape seed', 16: 'groundnuts',17: 'pulses', 18: 'citrus', 19: 'date palm', 20: 'grapes', 21: 'cotton', 22: 'cocoa', 23: 'coffee', 24: 'other perennial', 25: 'fodder grasses', 26: 'other annual'}


MIRCA2000_cropclassesStr={'1': 'wheat', '2': 'maize', '3': 'rice', '4': 'barley', '5': 'rye', '6': 'millet', '7': 'sorghum', '8': 'soybeans', '9': 'sunflower', '10': 'potatoes', '11': 'cassava', '12': 'sugarcane', '13': 'sugar beet', '14': 'oil palm', '15': 'rape seed', '16': 'groundnuts', '17': 'pulses', '18': 'citrus', '19': 'date palm', '20': 'grapes', '21': 'cotton', '22': 'cocoa', '23': 'coffee', '24': 'other perennial', '25': 'fodder grasses', '26': 'other annual'}


MIRCA2000_cropclassesStr2={'01': 'wheat', '02': 'maize', '03': 'rice', '04': 'barley', '05': 'rye', '06': 'millet', '07': 'sorghum', '08': 'soybeans', '09': 'sunflower', '10': 'potatoes', '11': 'cassava', '12': 'sugarcane', '13': 'sugar beet', '14': 'oil palm', '15': 'rape seed', '16': 'groundnuts', '17': 'pulses', '18': 'citrus', '19': 'date palm','20': 'grapes', '21': 'cotton', '22': 'cocoa', '23': 'coffee', '24': 'other perennial', '25': 'fodder grasses', '26': 'other annual'}



#-------- Function Crop Area Bar Plot

from functools import reduce

def CropAreaPlot(runval, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Crop ARea
    CA_data = pd.DataFrame(runval.CA_Data2.to_series())*1e-3# thousand km2
    CA = pd.DataFrame(runval.CA_Modeled.to_series())# thousand km2
    
    CA_list = [CA_data, CA]
    CA_join=reduce(join_dfs, CA_list)
    
    CA_group = CA_join.groupby(['clut','subcrop']).sum()
    CA_group.reset_index(inplace=True)
    CA_group = CA_group[CA_group['subcrop'] !='999999']
    CA_group['crop'] = CA_group['subcrop'].map(lambda x: MIRCA2000_cropclassesStr2[str(x[:-4])])
    
    
    BasinCA = CA_group.groupby(['clut','crop']).sum().reset_index()
    BasinCA = BasinCA.melt(id_vars=["clut","crop"], var_name="Type", value_name="Crop Area (000 km2)")
    BasinCA.dropna(inplace=True)
    BasinCA = BasinCA.sort_values(by=['Type','clut','Crop Area (000 km2)'])
    
    #remove zerost
    #BasinCA = BasinCA[BasinCA['Crop Area (000 km2)']!=0]
    
    #Rename Type
    data=BasinCA
    data = data.assign(Label=data.Type.map({'CA_Data2': "MIRCA Data", "CA_Modeled": "Estimate"}))
    
    # print to view
    BasinCA=data
    #print(BasinCA.head())
    
    #BasinCA['subcrop']= pd.to_numeric(BasinCA['subcrop'])
    #BasinCA.head()
    
    #-------------------------------- FIGURE
    # Crop Area
    import matplotlib.gridspec as gridspec
    
    #plt.figure(figsize=(15, 15))
    #G = gridspec.GridSpec(1, 1)
    #axes_1 = plt.subplot(G[0:, 0])
    
    import seaborn as sns
    clrs = ['red','blue'] #['grey' if (x < max(values)) else 'red' for x in values ]
    
    
    f = sns.catplot(data=BasinCA, x='crop', y='Crop Area (000 km2)',
                    hue='Label', col='clut',  kind='bar', palette=clrs, #sharey=True,
                    height=4, aspect=2, legend=False,ci=None)
        
    f.fig.set_size_inches(18, 5)
                    
    #f.fig.suptitle('Basin Crop Area (Thousand km2)', fontsize=20)
    f.fig.subplots_adjust(top=.9) #leave space
                    
                    
    f.set_xlabels('')
    f.set_xticklabels(rotation=90, fontsize=20)
                    
    f.set_ylabels("Crop Area ('000 $km^2$)", fontsize= 25)
    f.set_yticklabels(fontsize=25)
    f.set_titles(size=25)
                    
                    
    [plt.setp(ax.texts, text="") for ax in f.axes.flat] # remove the original texts # important to add this before setting titles
                    
    f.set_titles(col_template = '{col_name}')
                    
    plt.legend(loc='upper left', fontsize=20)
    
    
    f.savefig(savepath+'crop_area_basin_bar_graph.png', format='png', dpi=1200)
    print('saved in '+savepath)
                    
    #savepath = 'Images/'
    #f.savefig(savepath+'crop_area_bar_graph.pdf', format='pdf', dpi=1200)
    #print('saved in '+savepath)

    plt.show()


#------------Landuse Plots
# Land use
import gdx
import matplotlib.gridspec as gridspec

def LanduseTimeSeriesPlot (path,file,case, ira_data, rfa_data, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for i in range(len(file)):
        #print(case[i])
        runval = gdx.File(path+"/"+file[i])
        indata=runval;
        #casename = case[i]
    
        # Prepare data frame
        ira= monthsort(ToDF(runval.CA_time[:,:,:,0,:]).groupby(level=['time']).sum(),0) #thousand m2
        rfa= monthsort(ToDF(runval.CA_time[:,:,:,1,:]).groupby(level=['time']).sum(),0)
        ira.columns=['ira']
        rfa.columns=['rfa']
    
        CA_ir = ira_data.join(ira) # km2
        CA_rf = rfa_data.join(rfa) # km2
    
        # Correlation between data and modeled
        corr_ir = CA_ir.corr().iloc[0,1]
        corr_rf = CA_rf.corr().iloc[0,1]
    
        #print(corr_ir)
        #print(corr_rf)
    
        # Plot ############################################ TS
        plt.figure(figsize=(25, 5))
    
        G = gridspec.GridSpec(1,2)
        ax1 = plt.subplot(G[0, 0])
        ax2 = plt.subplot(G[0, 1])
    
        CA_ir.plot(figsize=(20,5), linewidth=5, color=['red','blue'], fontsize=15, ax=ax1, title="Irrigated Monthly Crop Area Basinwide (000 km2)")
        CA_rf.plot(figsize=(20,5), linewidth=5, color=['red','blue'], fontsize=15, ax=ax2, title="Rainfed Monthly Crop Area Basinwide (000 km2)")
    
        ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(corr_ir,2)), rotation=0, wrap=True, fontsize=15)
        ax2.text(ax2.get_xlim()[0], ax2.get_ylim()[1]+1, "R2: "+str(round(corr_rf,2)), rotation=0, wrap=True, fontsize=15)
    
        plt.legend(['Monfreda M3 Data','Estimate'])
        
        plt.savefig(savepath+'land_use_timeseries.png', format='png', dpi=1200)
        print('saved in '+savepath)
        
        plt.show()
    
    
    
        # Correlation of difference
        diff_corr_ir = CA_ir.diff().corr().iloc[0,1]
        diff_corr_rf = CA_rf.diff().corr().iloc[0,1]
    
        # Plot ############################################ DIFF TS
        plt.figure(figsize=(25, 5))
    
        G = gridspec.GridSpec(1,2)
        ax1 = plt.subplot(G[0, 0])
        ax2 = plt.subplot(G[0, 1])
    
        CA_ir.diff().plot(figsize=(20,5), linewidth=5, color=['red','blue'], fontsize=15, ax=ax1, title="Irrigated Month to Month Difference in CA" )
        CA_rf.diff().plot(figsize=(20,5), linewidth=5, color=['red','blue'], fontsize=15, ax=ax2, title="Rainfed Month to Month Difference in CA")
    
        ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(diff_corr_ir,2)), ha='left', rotation=0, wrap=True, fontsize=15)
        ax2.text(ax2.get_xlim()[0], ax2.get_ylim()[1]+1, "R2: "+str(round(diff_corr_rf,2)), ha='left', rotation=0, wrap=True, fontsize=15)
    
        
        
        plt.savefig(savepath+'diff_land_use_timeseries.png', format='png', dpi=1200)
        print('saved in '+savepath)
        
        plt.show()

        
        if case ==[]:
            casename ='default'
    
        col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
        CA_summary_stats_corr = pd.DataFrame(columns = col_names)
        
        CA_summary_stats_corr = CA_summary_stats_corr.append({'Case' : casename ,
                                                         'Estimated Variable' :  'Crop Area',
                                                         'Statistic': 'correlation',
                                                         'Value': corr_ir,
                                                         'Note':'ir'} , ignore_index=True)
        
        CA_summary_stats_corr =CA_summary_stats_corr.append({'Case' : casename ,
                                                        'Estimated Variable' :  'Crop Area',
                                                        'Statistic': 'correlation',
                                                        'Value': corr_rf,
                                                        'Note':'rf'} , ignore_index=True)
                                                         
        CA_summary_stats_corr =CA_summary_stats_corr.append({'Case' : casename ,
                                                         'Estimated Variable' :  'Crop Area',
                                                         'Statistic': 'correlation of diff',
                                                         'Value': diff_corr_ir,
                                                         'Note':'ir'} , ignore_index=True)
                                                         
        CA_summary_stats_corr =CA_summary_stats_corr.append({'Case' : casename ,
                                                        'Estimated Variable' :  'Crop Area',
                                                        'Statistic': 'correlation of diff',
                                                        'Value': diff_corr_rf,
                                                        'Note':'rf'} , ignore_index=True)
        stats = CA_summary_stats_corr
        return stats, ira, rfa



#----------- Data frame Joins
def join_dfs (ldf, rdf):
    return ldf.join(rdf, how='inner');



# Production Plotting ------------------------ no sugarcane
from functools import reduce

def plotProductionBar(path,file,runval, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Production (Monfreda)
    X_data = pd.DataFrame(runval.Production_Data.to_series())*1e-6 #million tons
    X = pd.DataFrame(runval.Production_Modeled.to_series())# million tons
    
    X_list = [X_data, X] # W is negative when out; sources are pos; sinks negative
    X_join=reduce(join_dfs, X_list)
    
    #X_join.head()
    
    X_group = X_join.groupby(['crop']).sum() # a little grown everywhere
    X_group.reset_index(inplace=True)
    X_group = X_group.dropna()
    
    BasinX = pd.melt(X_group, id_vars=["crop"], var_name="Type", value_name="Production (million tons)")
    BasinX.dropna(inplace=True)
    
    #remove zeros
    BasinX = BasinX[BasinX['Production (million tons)']!=0]
    BasinX = BasinX.sort_values(by=['Production (million tons)'])
    
    
    #Rename Type
    data=BasinX
    data = data.assign(Label=data.Type.map({'Production_Data': "Data", "Production_Modeled": "Estimate"}))
    
    # print to view
    BasinX=data
    #print(BasinX.head())
    
    #BasinCA['subcrop']= pd.to_numeric(BasinCA['subcrop'])
    #BasinCA.head()
    
    BasinX_ToPlot = BasinX[BasinX['crop']!='sugarcane']
    SugarcaneX_ToPlot = BasinX[BasinX['crop']=='sugarcane']
    

    
    #-----------------------------------------
    #Production Data
    import matplotlib.gridspec as gridspec
    
    plt.figure(figsize=(15, 5))
    
    #axes_1 = plt.subplot(111)
    
    import seaborn as sns
    clrs = ['blue','red'] #['grey' if (x < max(values)) else 'red' for x in values ]
    
    
    f = sns.catplot(data=BasinX_ToPlot, x='crop', y='Production (million tons)',
                       hue='Label',  kind='bar', palette =clrs, #ax=axes_1,
                       height=10, aspect=1, legend=False, legend_out=True)
        
    f.fig.set_size_inches(15,5)
    f.fig.suptitle('Basin Production (million tons)', fontsize=20)
    f.fig.subplots_adjust(top=.9) #leave space
    
    f.set_xlabels('Crops', fontsize= 20)
    f.set_xticklabels(rotation=45, fontsize=15)
                       
    f.set_ylabels('Production(million tons)', fontsize= 15)
    f.set_yticklabels(fontsize=15, rotation=0)
    f.set_titles(size=20)
                       
    plt.legend(loc='upper left', fontsize=15)
                       
    plt.ylim([0,20])
    
    plt.savefig(savepath+'production_basinwide_bar_graph.png', format='png', dpi=1200)
    print('saved in '+savepath)
                       
    plt.show()
    
    return BasinX
    
    # -----------------
    # Sugarcane Data
    plt.figure(figsize=(2, 5))
    
    #axes_1 = plt.subplot(111)
    
    import seaborn as sns
    clrs = ['blue','red'] #['grey' if (x < max(values)) else 'red' for x in values ]
    
    
    f = sns.catplot(data=SugarcaneX_ToPlot, x='crop', y='Production (million tons)',
                       hue='Label',  kind='bar', palette =clrs, #ax=axes_1,
                       height=10, aspect=1, legend=False, legend_out=True)
        
    f.fig.set_size_inches(2,5)
    f.fig.suptitle('Basin Production (million tons)', fontsize=20)
    f.fig.subplots_adjust(top=.9) #leave space
    
    f.set_xlabels('Crops', fontsize= 20)
    f.set_xticklabels(rotation=45, fontsize=15)
                       
    f.set_ylabels('Production(million tons)', fontsize= 15)
    f.set_yticklabels(fontsize=15, rotation=0)
    f.set_titles(size=20)
                       
    plt.legend(loc='upper left', fontsize=15)
                       
    plt.ylim([0,70])
    
    plt.savefig(savepath+'sugarcane_production_basinwide_bar_graph.png', format='png', dpi=1200)
    print('saved in '+savepath)
                       
                       
    plt.show()
    
    
#-----------

def plotGWHeadScatter(runval, casename, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    gw_data = pd.DataFrame(runval.H.to_series().dropna())*1e3
    gw = pd.DataFrame(runval.Head_Data.to_series().dropna())

    X_list = [gw_data, gw]
    X_join=reduce(join_dfs, X_list) 


    df=X_join

    var = 'time'
    obs = 'Head_Data'
    est = 'H'
    Title = 'Heads (m)'
    colorstyle = 'r--'

    # Plot
    import seaborn as sns
    plt.figure(figsize=(10, 5))

    G = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(G[0, 0])
    df1 = df
    mark_size=150
    ScatterPlotObsvEst(df1, var, obs, est, Title, ax1, colorstyle, mark_size)
    
    plt.savefig(savepath+'heads_scatter.png', format='png', dpi=1200)
    print('saved in '+savepath)    
    
    plt.show()
    
    col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
    head_summary_stats_slope  = pd.DataFrame(columns = col_names)
    return getSummaryStats(df, var, obs, est, head_summary_stats_slope, 'Heads_by_month (vs. data)')
    
 ###



def plotMeanBasinHeadTS(runval,casename, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    H = pd.DataFrame(runval.H.to_series().dropna())*1e3
    H_data = pd.DataFrame(runval.Head_Data.to_series().dropna())

    X_list = [H_data, H] 
    X_join=reduce(join_dfs, X_list) 

    df=X_join

    # Correlation
    corr = monthsort(df.groupby('time').mean(),0).corr().iloc[0,1]

    # Plot ############################################
    plt.figure(figsize=(10, 5))
    G = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(G[0, 0])
    monthsort(df.groupby('time').mean(),0).plot(figsize=(10,5), linewidth=5, color=['red','blue'], fontsize=20, ax=ax1, title="Mean Basin Heads")
    ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(corr,2)), rotation=0, wrap=True, fontsize=15)
    
    plt.savefig(savepath+'mean_basin_head_timeseries.png', format='png', dpi=1200)
    print('saved in '+savepath)  
    
    plt.show()
    

    # Correlation of difference
    diff_corr = monthsort(df.groupby('time').sum(),0).diff().corr().iloc[0,1]
    
    '''
     # Plot ############################################
    plt.figure(figsize=(10, 5))
    G = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(G[0, 0])   
    monthsort(df.groupby('time').mean(),0).diff().plot(figsize=(10,5), linewidth=5, color=['red','blue'], fontsize=20, ax=ax1, title="Month to Month Diff of Mean Basin Heads")      
    ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(diff_corr,2)), ha='left', rotation=0, wrap=True, fontsize=15) 
    plt.show()
    '''
    
    col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
    head_summary_stats_corr = pd.DataFrame(columns = col_names)
    
    # Store summary stats
    head_summary_stats_corr = head_summary_stats_corr.append({'Case' : casename , 
                         'Estimated Variable' :  "Basin Head",
                         'Statistic': 'correlation',
                         'Value': corr,
                         'Note':'(vs. data)'                
                         } , ignore_index=True)

    head_summary_stats_corr =head_summary_stats_corr.append({'Case' : casename , 
                         'Estimated Variable' :  'Basin Head',
                         'Statistic': 'correlation of diff',
                         'Value': diff_corr,
                         'Note':'(vs. data)'                
                         } , ignore_index=True)

    return df, head_summary_stats_corr
    
#------------

def plotrunoff(runval,indata, savepath):
    import pandas as pd
    import numpy as np
    import os
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    path = os.getcwd()
    filename = 'All_Krishna_Stations_Monthly_Discharge_km3permo_2000_2010_ForGAMS.xlsx'

    xl = pd.ExcelFile(path+"/"+filename) 
    station_loc_df = xl.parse("UniqueForGAMS (2)")
    station_loc_df.head(5)

    # Stations
    stationlist = station_loc_df.iloc[:,[0,1,2,3,4,5,18]]
    stationlist = stationlist.rename(columns={'Row': 'r', 'Col': 'c'})

    available_stations = list(stationlist.loc[:,'Station'])
    available_stations.sort()

    # Prepare Data
    R_Data = ToDF(indata. R_Data).reset_index().astype({'r':'int64','c':'int64'})
    R_Model = ToDF(runval.R_Modeled).reset_index().astype({'r':'int64','c':'int64'})
    merged1 = reduce(lambda left,right: pd.merge(left,right,on=['r','c','time']), [R_Data,R_Model])
    MonthlyRunoff = reduce(lambda left,right: pd.merge(left,right,on=['r','c',]), [merged1,stationlist])

    MonthlyRunoff['time']=MonthlyRunoff['time'].astype('category')
    sorter = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    MonthlyRunoff.time.cat.set_categories(sorter, inplace=True)

    import calendar
    mo_convert = dict((v,k) for k,v in enumerate(calendar.month_abbr))
    MonthlyRunoff['mo_num'] = MonthlyRunoff['time'].map(mo_convert)
    MonthlyRunoff['mo_num'].astype('int64');

    #print(available_stations)

    # Get runoff Data
    R_Data=ToDF(indata. R_Data)
    #R_Data.index = R_Data.index.droplevel(level='Rsource')
    R_Data_format=R_Data.unstack()

    # Get Modeled Data
    Qout_Mod=ToDF(runval.R_Modeled)
    Qout_Mod = Qout_Mod.unstack()

    # Select guage locations based on supplied data
    select = np.transpose(Qout_Mod)[R_Data_format.index[0:len(R_Data_format)][0:len(R_Data_format)]]
    Qout_Mod_guages= np.transpose(select)
    Qout_Mod_guages

    # Concatenate data and model selection
    runoff_guages = pd.concat([R_Data_format.sum(axis = 1), Qout_Mod_guages.sum(axis=1)], axis=1)
    runoff_guages.reset_index(inplace=True)
    runoff_guages = runoff_guages.astype({'r':'int64','c':'int64'})
    runoff_guages.rename(columns={0:'R_Data',1:'R_Modeled'},inplace=True)

    dfs = [stationlist, runoff_guages]
    AnnualRunoff = reduce(lambda left,right: pd.merge(left,right,on=['r','c']), dfs)
    #print(AnnualRunoff.head())

    AnnualRunoffForPlot = AnnualRunoff.iloc[:,[0,3,7,8]]
    AnnualRunoffForPlot.head()


    AnnRunOff = pd.melt(AnnualRunoffForPlot, id_vars=["Station","Loc"], var_name="Type", value_name="Runoff Value (km3)")
    #print(AnnRunOff.head())

    NameDict={'R_Data':'Data','R_Modeled':'Estimate'}
    AnnRunOff['label']=AnnRunOff['Type'].map(lambda x: NameDict[x])


    test = AnnRunOff[AnnRunOff['Runoff Value (km3)']>5]
    sel = np.unique(test.Station)


    # Figure Runoff
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np


    plt.figure(figsize=(10, 8))

    #Stations
    st = ['Vijayawada']
    data=MonthlyRunoff

    #Data
    data['time']=MonthlyRunoff['time'].astype('category')
    sorter = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    MonthlyRunoff.time.cat.set_categories(sorter, inplace=True)

    for i in range(1,len(st)+1):
        subset = data[data.Station==st[i-1]]
        plt.plot(subset['mo_num'], subset['R_Data'], marker='o', linestyle='', markersize=12, color='r')
        plt.plot(subset['mo_num'], subset['R_Modeled'], marker='', linestyle='-', color='b')
        plt.title(st[i-1])
        plt.ylim([0,15])
        plt.xticks(np.arange(1,12), sorter, rotation=0, fontsize=16)
        plt.xlabel('Month')
        plt.ylabel('Runoff Value (km3)')



    plt.legend(['Data','Estimate']) 
    
    
    plt.savefig(savepath+'Fig9_runoff_basin_timeseries.jpeg', format='jpeg', dpi=1200)
    print('saved in '+savepath) 
    

    plt.show()
    return subset
    
    
#function to plot
#------------ plot GW Head

def plotGWDepletionScatter(runval, casename, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    X_data = pd.DataFrame(runval.gwd.to_series().dropna())
    X = pd.DataFrame(runval.depletion.to_series().dropna())*-1# million tons

    X_list = [X_data, X] # W is negative when out; sources are pos; sinks negative
    X_join=reduce(join_dfs, X_list) 

    df=X_join.reset_index()[['gwd','depletion']]

    var='GW Depletion'
    obs='gwd'
    est='depletion'

    # Calculate statistics
    x = df[obs].values
    y = df[est].values

    #print(len(x))

    popt, pcov = curve_fit(func, x, y) 
    slope = popt[0]

    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    # Plot
    plt.figure(figsize=(10, 5))

    G = gridspec.GridSpec(1,1)
    ax = plt.subplot(G[0, 0])
    scatter = ax.scatter(df[obs], df[est], s=100)

    plt.xlabel('Observed', size=14)
    plt.ylabel('Estimate', size=14)
    plt.tick_params(labelsize=14)

    plt.title("Scatter Plot Obs vs Est for Depletion", size=16)

    abline(slope,0, 'r--', 'Linear Fit:  y='+str(round(slope,2))+"x")
    abline(1,0, 'k--',"1:1")
    
    plt.savefig(savepath+'gw_depletion_scatter.png', format='png', dpi=1200)
    print('saved in '+savepath) 
    
    plt.show()
    
    col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
    gw_summary_stats_slope  = pd.DataFrame(columns = col_names)

    return getSummaryStats(df, var, obs, est, gw_summary_stats_slope, 'Depletion (vs. data)')

#---------
def ROM(runval, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    
    #ROM
    plt.rcParams.update({'font.size':16})

    monthlabel=['J','F','M','A','M','J','J','A','S','O','N','D'];

    indata = runval
    
    #Surrogate
    BasinArea = indata.BasinArea.values
    BasinArea_km2 = BasinArea/1e6
    mm2km = 1e-6;

    dfE = pd.DataFrame(monthsort(runval.ET_Modeled.to_series().groupby(level=['time']).sum(),0))

    aet_md_mm=[28,16,12,11,22,53,79,93,87,81,65,46]

    #ROM
    dfErom = pd.DataFrame(monthsort(runval.ET_ROM_pixel.to_series().groupby(level=['time']).sum(),0))

    # Dataframe
    dfE = dfE.assign(MD_Data=np.array(aet_md_mm)*mm2km*BasinArea_km2)
    dfE = dfE.assign(ET_Data=runval.ET_Data.sum(axis=1).sum(axis=0))
    dfE = dfE.assign(ET_ROM=runval.ET_ROM_pixel.sum(axis=1).sum(axis=0))
    dfE.head()


    # Figure ROM Vs Modeled
    plt.figure(figsize=(20, 10))
    G = gridspec.GridSpec(2,3)

    axes_1 = plt.subplot(G[0, 0])

    #AET
    data = dfE[['ET_ROM','ET_Modeled','ET_Data']]
    data = (data/BasinArea_km2)*1e6 

    plt.plot(data, marker='o', linestyle='solid', linewidth=1, markersize=8)
    plt.legend(['Surrogate','Estimate','Data'])
    plt.title('Evapotranspiration (mm)')
    plt.ylabel('Depth (mm)')
    plt.ylim([0,120])
    plt.xticks(np.arange(0,12), monthlabel, rotation=0, fontsize=16)


    # Soil Moisture
    axes_1 = plt.subplot(G[0, 1])
    dfS = pd.DataFrame(monthsort(runval.dS_Modeled.to_series().groupby(level=['time']).sum(),0))
    dS_ROM = pd.DataFrame(monthsort(runval.dS_ROM_pixel.to_series().groupby(level=['time']).sum(),0))
    dfS = dfS.assign(dS_ROM=dS_ROM)

    data = dfS
    data = (data/BasinArea_km2)*1e6 

    plt.plot(data, marker='o', linestyle='solid', linewidth=1, markersize=8)
    plt.title('Change in Soil Moisture (mm)')
    plt.ylabel('Depth (mm)')
    plt.legend(['Estimate','Surrogate'], fontsize=14)
    plt.ylim([-60,60])
    plt.xticks(np.arange(0,12), monthlabel, rotation=0, fontsize=16)


    # Recharge
    axes_1 = plt.subplot(G[0, 2])
    dfW = pd.DataFrame(monthsort(runval.W_Modeled.to_series().groupby(level=['time']).sum(),0))
    dfWrom = pd.DataFrame(monthsort(runval.W_ROM_pixel.to_series().groupby(level=['time']).sum(),0))
    dfP = pd.DataFrame(monthsort(runval.P_Modeled.to_series().groupby(level=['time']).sum(),0))*0.15 # 20percent precip

    dfW = dfW.assign(W_ROM=np.array(dfWrom))
    dfW = dfW.assign(W_PP=np.array(dfP))

    data=dfW
    data = (data/BasinArea_km2)*1e6 

    plt.plot(data, marker='o', linestyle='solid', linewidth=1, markersize=8)
    plt.title('Recharge (mm)')
    plt.ylabel('Depth (mm)')
    plt.legend(['Estimate','Surrogate','15% of Precip'], fontsize=14)
    plt.ylim([0,30])
    plt.xticks(np.arange(0,12), monthlabel, rotation=0, fontsize=16)
    
    
    plt.savefig(savepath+'et_soil_moisture_recharge_timeseries.png', format='png', dpi=1200)
    print('saved in '+savepath) 

    plt.show()

#--------
# Function Yield vs ET: plot = plotYvsET(crop,file)

def plotYvsET(crop, file):
    for i in range(len(file)):
        print(file[i])
        runval = gdx.File(path+file[i])
        indata=runval;

        scs = list(runval.subcrop.data)

        #YIELD
        YieldMod = pd.DataFrame(runval.Yield_Modeled.to_series())# tons per ha
        #print(YieldMod.head())

        #WATER
        et = runval.ET.sum('time')
        ET = pd.DataFrame(et.to_series())

        #print(ET.head())

        #SEPARATE IR AND RF
        df=YieldMod.reset_index()
        irY = df[df['clut']=='ir'].drop(labels='clut', axis=1).dropna()
        rfY = df[df['clut']=='rf'].drop(labels='clut', axis=1).dropna()

        df=ET.reset_index()
        irET =df[df['lut']=='ir'].drop(labels='lut', axis=1).dropna()
        rfET =df[df['lut']=='ir'].drop(labels='lut', axis=1).dropna()


        Y_ET_ir = irY.merge(irET, on=['r','c','subcrop'], how='outer')
        Y_ET_rf = rfY.merge(rfET, on=['r','c','subcrop'], how='outer')

        sugarcane = [['120112']]
        rice = ['030710','030711','031204' ]
        sorg = ['070711', '071204', '070510']
        otherann = ['260610', '261103']


        #STATS
        df1 = Y_ET_ir.dropna()
        df2 = Y_ET_rf.dropna()

        #print(df1)
        #print(df2)

        print("IR ET Mean:" + str(df1[df1['subcrop'].isin(crop)]['ET'].mean()))
        print("IR Yield Mean:" +str(df1[df1['subcrop'].isin(crop)]['Yield_Modeled'].mean()))
        print("")
        print("RF ET Mean:" + str(df2[df2['subcrop'].isin(crop)]['ET'].mean()))
        print("RF Yield Mean:" +str(df2[df2['subcrop'].isin(crop)]['Yield_Modeled'].mean()))

        #PLOT
        #crop = rice
        plt.scatter(df1[df1['subcrop'].isin(crop)]['ET'], df1[df1['subcrop'].isin(crop)]['Yield_Modeled'], color='b', alpha=0.5,)
        plt.scatter(df2[df2['subcrop'].isin(crop)]['ET'], df2[df2['subcrop'].isin(crop)]['Yield_Modeled'], color='cyan', alpha=0.2)
        #plt.legend()
        #plt.xlim([100,1000])
        plt.ylabel('Yield (tons/ha)')
        plt.xlabel('ET (mm)')
        plt.show()
    
    
#----------- Yield
# Function Y vs ET Data: iryet, rfyet = YvsETData(runval)
def YvsETData(runval):

        scs = list(runval.subcrop.data)

        #YIELD
        YieldMod = pd.DataFrame(runval.Yield_Modeled.to_series())# tons per ha
        #print(YieldMod.head())

        #WATER
        et = runval.ET.sum('time')
        ET = pd.DataFrame(et.to_series())

        #print(ET.head())

        #SEPARATE IR AND RF
        df=YieldMod.reset_index()
        irY = df[df['clut']=='ir'].drop(labels='clut', axis=1).dropna()
        rfY = df[df['clut']=='rf'].drop(labels='clut', axis=1).dropna()

        df=ET.reset_index()
        irET =df[df['lut']=='ir'].drop(labels='lut', axis=1).dropna()
        rfET =df[df['lut']=='ir'].drop(labels='lut', axis=1).dropna()


        Y_ET_ir = irY.merge(irET, on=['r','c','subcrop'], how='outer')
        Y_ET_rf = rfY.merge(rfET, on=['r','c','subcrop'], how='outer')

        #STATS
        iryet = Y_ET_ir.dropna()
        rfyet = Y_ET_rf.dropna()
        
        return iryet, rfyet
    
    
    
    
# function GuagePlot
#-------------------------------------------------
def GuagePlot(df):
    fig, (ax1) = plt.subplots(1, 1, sharey=False,  figsize=(6,4))
    
    df.R_Data.plot(ax=ax1,  marker='o', linestyle='', markersize=11, color='g')
    df.Qcout.plot(ax = ax1, color='g')
    
    plt.legend(['R Data', 'R Model'], loc='upper left', ncol = 1)
    plt.xlabel('Months')
    plt.xlim([0,12])
    
    plt.ylabel('Volume (cu. km)')
    plt.title('Runoff')
    
    plt.show()

    
#-------
# Function Rel Yield vs Rel Water DAta: iryw, rfyw = RelYield_vRelWater_Data(runval):  
def RelYield_vRelWater_Data(runval):  
    # CREATE YIELD DATAFRAME
    Ya = pd.DataFrame(runval.Yield_Modeled.to_series()) #n,cl,sc
    Yp = pd.DataFrame(runval.Yin.to_series())

    Yields_df= Ya.merge(Yp, on=['r','c','clut','subcrop'], how='outer')
    Yields_df['Ya_Yp'] = Yields_df['Yield_Modeled']/Yields_df['Yin'] 
    Yields_df.dropna()

    #print(Yields_df.head())

    # CREATE WATER DATAFRAME
    PETf = pd.DataFrame(runval.kcEtof.to_series()) # n,t,l,sc
    AETf = pd.DataFrame(runval.ETf.to_series()) # n,t,l,sc

    EvapoTrans_df= AETf .merge(PETf , on=['r','c','time','lut','subcrop'], how='outer')
    SeasET_df = EvapoTrans_df.groupby(['r','c','lut','subcrop']).sum()
    SeasET_df['AET_PET'] = SeasET_df['ETf']/SeasET_df['kcEtof'] 
    SeasET_df.dropna().head()
    SeasET_df.rename_axis(['r','c','clut','subcrop'], inplace=True)

    #print(SeasET_df.head())

    # CREATE COMBINED YIELD-WATER DATAFRAME
    YieldWater = Yields_df.merge(SeasET_df, on=['r','c','clut','subcrop'], how='outer').dropna()
    #print(YieldWater.head())

    # SPLIT IRRIGATED AND RAINFED
    df = YieldWater.reset_index()

    #print(df)

    IR_YW = df[df['clut']=='ir'].drop('clut', axis=1)
    RF_YW = df[df['clut']=='rf'].drop('clut', axis=1)

    # SELECT CROP and PLOT
    #crop = rice
    iryw = IR_YW.dropna()
    rfyw = RF_YW.dropna()
    
    return iryw, rfyw

# Main function for Yield Water Relation plots

def PlotYielWaterRelation(ax1,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var1, var2):
    num = len(subcroplist)

    ax1.scatter(kykset_ir[kykset_ir['subcrop'].isin(subcroplist)][var1], kykset_ir[kykset_ir['subcrop'].isin(subcroplist)][var2], color=color_ir, alpha=0.5, label='irrigated')
    ax1.scatter(kykset_rf[kykset_rf['subcrop'].isin(subcroplist)][var1], kykset_rf[kykset_rf['subcrop'].isin(subcroplist)][var2], color=color_rf, alpha=0.2, label='rainfed')

##


def Plot_RelativeYield_For_Sugarcane_Sorghum_OtherAnnual(runval, i, var, var2, savepath):
    
        import os
        if not os.path.exists(savepath):
            os.makedirs(savepath)


        kykset_ir,kykset_rf = RelYield_vRelWater_Data(runval)

        plt.figure(figsize=(25, 5))
        G = gridspec.GridSpec(1,3)

        # Plot 1
        ax1 = plt.subplot(G[0, 0])

        subcroplist = ['120112']#sugarcane
        color_ir='b'
        color_rf='cyan'

        PlotYielWaterRelation(ax1,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)

        ax1.legend()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('Ya/Yp')
        plt.xlabel('AET/PET')
        plt.title('Sugarcane')

        # Plot2
        ax2 = plt.subplot(G[0, 1]) 

        subcroplist =  ['070711', '071204', '070510']#sorghum
        color_ir='forestgreen'
        color_rf='lime'

        PlotYielWaterRelation(ax2,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)

        ax2.legend()
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.ylabel('Ya/Yp')
        plt.xlabel('AET/PET')
        plt.title('Sorghum')

        # Plot 3
        ax3 = plt.subplot(G[0, 2])

        subcroplist = ['260610', '261103']
        color_ir='red'
        color_rf='coral'

        PlotYielWaterRelation(ax3,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)

        ax3.legend()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('Ya/Yp')
        plt.xlabel('AET/PET')
        plt.title('Other Annual')
        
        
        plt.savefig(savepath+'relative_yield_water_scatterplot_sc.png', format='png', dpi=1200)
        print('saved in '+savepath)    
        
        plt.show()

#----
#Main Short


def Plot_RelativeYield_For_Rice_Wheat_Pulses(runval, i, var, var2, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    kykset_ir,kykset_rf = RelYield_vRelWater_Data(runval)
    
    plt.figure(figsize=(25, 5))
    G = gridspec.GridSpec(1,3)
    
    # Plot 1
    ax1 = plt.subplot(G[0, 0])
    
    subcroplist = ['030710','030711','031204'] #rice
    color_ir='darkorange'
    color_rf='peachpuff'
    
    PlotYielWaterRelation(ax1,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)
    
    ax1.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('Ya/Yp')
    plt.xlabel('AET/PET')
    plt.title('Rice')

    # Plot2
    ax2 = plt.subplot(G[0, 1]) 
    
    subcroplist = ['010711','011204','011206']#wheat
    color_ir='chocolate'
    color_rf='sienna'
    
    PlotYielWaterRelation(ax2,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)
    
    ax2.legend()
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.ylabel('Ya/Yp')
    plt.xlabel('AET/PET')
    plt.title('Wheat')

    # Plot 3
    ax3 = plt.subplot(G[0, 2])
    
    subcroplist = ['171002'] #pulses
    color_ir='purple'
    color_rf='violet'
    
    PlotYielWaterRelation(ax3,kykset_ir,kykset_rf,subcroplist, color_ir, color_rf, i, var, var2)

    ax3.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('Ya/Yp')
    plt.xlabel('AET/PET')
    plt.title('Pulses')
    
    
    plt.savefig(savepath+'relative_yield_water_scatterplot_rice.png', format='png', dpi=1200)
    print('saved in '+savepath)  
    
    plt.show()


#------
# Function croplandET plot
def croplandET(runval,CA_dat, savepath): 
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    indata=runval;
    BasinArea = indata.BasinArea.values
    BasinArea_km2 = BasinArea/1e6
    
    # Cropland Modeled
    ira= monthsort(ToDF(runval.CA_time[:,:,:,0,:]).groupby(level=['time']).sum(),0) #thousand m2
    rfa= monthsort(ToDF(runval.CA_time[:,:,:,1,:]).groupby(level=['time']).sum(),0)
    ira.columns=['ira']
    rfa.columns=['rfa']
    CA_mod = ira.join(rfa)

    colorpal=['lightskyblue','seagreen','khaki'];

    monthlabel=['J','F','M','A','M','J','J','A','S','O','N','D'];

    #ET Plot and CA -------------------

    plt.figure(figsize=(20,5))
    G = gridspec.GridSpec(1,3)
    m2Tokm2 =1e-6 

    axes_1 = plt.subplot(G[0, 0])

    #Parameter ET_byCompartment_inTime(m,l);
    #ET_byCompartment_inTime(m,l) = sum(n$Domain(n), (sum(sc, ETf.l(n,m,l,sc)$(sl(l,sc) and season(sc,m)) *mmTokm *A(n)*m2Tokm2 ))

    ET_vol_km3 = ToDF(runval.ET_byCompartment_inTime)
    ET_vol_km3 = ET_vol_km3.reset_index()

    ET_vol_km3['ET_mm'] = ET_vol_km3['ET_byCompartment_inTime']/(BasinArea*m2Tokm2)*1e6
    ET_vol_km3

    IRET = ET_vol_km3[ET_vol_km3['lut']=='ir']
    RFET = ET_vol_km3[ET_vol_km3['lut']=='rf']
    NCET = ET_vol_km3[ET_vol_km3['lut']=='nc']



    # Modeled Water Balance Parts in Dataframe- Month by Compartment in mm
    #E_mod = CompartmentTS(runval.ETf.to_series()) # can't just use this because of the summing of crops that are not in season
    #data = E_mod
    #data = (E_mod*1e-3 )#/(BasinArea*m2Tokm2))*1e6




    axes_1.stackplot(np.array(range(1,13)),np.array(IRET['ET_mm']), np.array(RFET['ET_mm']), np.array(NCET['ET_mm']),colors=colorpal)
    plt.title('Estimated Evapotranspiration (mm)',fontsize=16)
    plt.ylabel('ET depth (mm)', fontsize=16)
    plt.xlim([1,12])
    plt.legend(['Irrigated','Rainfed','Noncrop'], loc='upper left',fontsize=16)
    plt.xlabel('Months',fontsize=16)
    plt.xticks(np.arange(1,13), monthlabel, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([0,120])

    

    #--------ARea

    axes_2 = plt.subplot(G[0, 1])

    data = CA_mod*1000/(BasinArea*m2Tokm2)*100; #km2 by km2
    data['nca']=100-data['ira']-data['rfa']
    axes_2.stackplot(np.array(range(1,13)),np.array(data['ira']), np.array(data['rfa']), np.array(data['nca']), colors=colorpal)
    plt.title('Estimated Crop Area (Percent Basin)',fontsize=16)
    plt.ylabel('Percent (%)',fontsize=16)
    plt.ylim([0,100])
    plt.xlim([1,12])
    plt.xticks(np.arange(1,13), monthlabel, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    #plt.legend(['Irrigated','Rainfed', 'Noncrop'], loc='upper left')
    plt.xlabel('Months',fontsize=16)



    axes_3 = plt.subplot(G[0, 2])

    data = CA_dat/(BasinArea*m2Tokm2)*100;
    data['nca']=100-data['ira']-data['rfa']

    axes_3.stackplot(np.array(range(1,13)),np.array(data['ira']), np.array(data['rfa']), np.array(data['nca']), colors=colorpal)
    plt.title('MIRCA Crop Data Area (Percent Basin)',fontsize=16)
    plt.ylabel('Percent (%)',fontsize=16)
    plt.ylim([0,100])
    plt.xlim([1,12])
    plt.xticks(np.arange(1,13), monthlabel, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    #plt.legend(['Irrigated','Rainfed', 'Noncrop'], loc='upper left')
    plt.xlabel('Months',fontsize=16)

    plt.savefig(savepath+'et_by_landuse_timeseries.png', format='png', dpi=1200)
    print('saved in '+savepath) 

    plt.show()

#------
def ETbyCrop(runval,CA_dat, savepath):  
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    dom = runval.Domain.to_series()
    
    indata=runval;
    BasinArea = indata.BasinArea.values
    BasinArea_km2 = BasinArea/1e6
    
    # Cropland Modeled
    ira= monthsort(ToDF(runval.CA_time[:,:,:,0,:]).groupby(level=['time']).sum(),0) #thousand m2
    rfa= monthsort(ToDF(runval.CA_time[:,:,:,1,:]).groupby(level=['time']).sum(),0)
    ira.columns=['ira']
    rfa.columns=['rfa']
    CA_mod = ira.join(rfa)

    colorpal=['lightskyblue','seagreen','khaki'];

    monthlabel=['J','F','M','A','M','J','J','A','S','O','N','D'];

    #ET Plot and CA -------------------

    plt.figure(figsize=(20,5))
    G = gridspec.GridSpec(1,3)
    m2Tokm2 =1e-6 

    axes_1 = plt.subplot(G[0, 0])

    #Parameter ET_byCompartment_inTime(m,l);
    #ET_byCompartment_inTime(m,l) = sum(n$Domain(n), (sum(sc, ETf.l(n,m,l,sc)$(sl(l,sc) and season(sc,m)) *mmTokm *A(n)*m2Tokm2 ))

    ET_vol_km3 = ToDF(runval.ET_byCompartment_inTime)
    ET_vol_km3 = ET_vol_km3.reset_index()

    ET_vol_km3['ET_mm'] = ET_vol_km3['ET_byCompartment_inTime']/(BasinArea*m2Tokm2)*1e6
    ET_vol_km3

    IRET = ET_vol_km3[ET_vol_km3['lut']=='ir']
    RFET = ET_vol_km3[ET_vol_km3['lut']=='rf']
    NCET = ET_vol_km3[ET_vol_km3['lut']=='nc']



    # Modeled Water Balance Parts in Dataframe- Month by Compartment in mm
    #E_mod = CompartmentTS(runval.ETf.to_series()) # can't just use this because of the summing of crops that are not in season
    #data = E_mod
    #data = (E_mod*1e-3 )#/(BasinArea*m2Tokm2))*1e6




    axes_1.stackplot(np.array(range(1,13)),np.array(IRET['ET_mm']), np.array(RFET['ET_mm']), np.array(NCET['ET_mm']),colors=colorpal)
    plt.title('Estimated Evapotranspiration (mm)',fontsize=16)
    plt.ylabel('ET depth (mm)', fontsize=16)
    plt.xlim([1,12])
    plt.legend(['Irrigated','Rainfed','Noncrop'], loc='upper left',fontsize=16)
    plt.xlabel('Months',fontsize=16)
    plt.xticks(np.arange(1,13), monthlabel, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([0,120])

    

    #--------Irrigated

    axes_2 = plt.subplot(G[0, 1])

    df = ToDF(runval.ETf).reset_index().groupby(['time','lut','subcrop']).sum()/dom.sum()
    df = df.reset_index()
    
    var = 'ETf'
    
    ir = df[df['lut']=='ir']
    ir = ir[ir[var]>0]
    
    ir2 = SubcropCodeToCropName(ir)
    ir3 = ir2.groupby(['time','crop']).sum().reset_index().dropna()
    ir4 = ir3[['time','crop',var]]
    pivot_ir = ir4.pivot(index='time', columns='crop', values=var)
    
    pivot_ir.plot(kind='area', stacked='True', ax=axes_2)
    plt.ylim([0,120])
    plt.title('Estimated Irrigated ET by crop (mm)', fontsize=16)
    axes_2.legend(loc='upper left', fontsize=10)
    

    
    #--------Rainfed

    axes_3 = plt.subplot(G[0, 2])

    rf = df[df['lut']=='rf']
    rf = rf[rf[var]>0]
    rf2 = SubcropCodeToCropName(rf)
    rf3 = rf2.groupby(['time','crop']).sum().reset_index().dropna()
    rf4 = rf3[['time','crop',var]]
    pivot_rf = rf4.pivot(index='time', columns='crop', values=var)
    pivot_rf.plot(kind='area', stacked='True', ax=axes_3)
    plt.ylim([0,120])
    plt.title('Estimated Rainfed ET by crop (mm)', fontsize=16)
    axes_3.legend(loc='upper left', fontsize=10)
    
    plt.savefig(savepath+'et_bycrop.png', format='png', dpi=1200)
    print('saved in '+savepath) 

    plt.show()


    

# function as percent
#------------------ As Percent
from numbers import Number
def as_percent(v, precision):  
    """Convert number to percentage string."""
    if isinstance(v, Number):
        return "{{:{}%}}".format(precision).format(v)
    else:
        raise TypeError("Numeric type required")


# function for Basemap
#------------------ Make Basemap

from mpl_toolkits.basemap import Basemap


def MakeBasemap(ax):
    # Locate a bounding box
    ll_long = 73.4; ll_lat = 12.9; ur_long = 81.1; ur_lat = 19.6
    # Create Basemap
    
    map = Basemap(llcrnrlon=ll_long,llcrnrlat=ll_lat,urcrnrlon=ur_long,urcrnrlat=ur_lat, ax=ax)
    #             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
    
    # Bring In Shapefile with Parallels and Meredians
    shp_info_basin = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/KrishnaBasin', 'KrishnaBasin', drawbounds = True, color ='k', linewidth=1.0)
    gridshp_info = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/Krishna_Fishnet', 'Krishna_Fishnet', drawbounds = True, color='grey', linewidth=0.5)
    
    parallels = np.arange(13,20,1) # make latitude lines ever 1 degrees
    meridians = np.arange(70,84,1) # make longitude lines every 1 degrees from 95W to 70W
    
    # labels = [left,right,top,bottom]   
    map.drawparallels(parallels,labels=[True,False,False,False],fontsize=10)
    map.drawmeridians(meridians,labels=[False,False,False,True],fontsize=10)
    
    return map

#------------------ Plot Data
import pylab

def PlotData(Vec, pos, time, fig, ax, title, cbarformat):
    params = {
         'axes.labelsize': 'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
    pylab.rcParams.update(params)
    method = 'nearest' #'bilinear'
    MaxVec = Vec.max().max()
    MinVec = Vec.min().min()
    #pos=i*5+posnum; #depends on the number of vectors displayed
    Data = np.array(Vec[[str(i+1)]])
    cax = ax[pos].imshow(Data, interpolation=method, cmap=plt.cm.jet, vmin=MinVec, vmax=MaxVec)
    ax[pos].set_title(title,fontsize=15)
    x = np.linspace(MinVec, MaxVec, 5 , endpoint=True)
    cbar = fig.colorbar(cax, ax=ax[pos],ticks=x, orientation='vertical',fraction=0.046, pad=0.04, format=cbarformat)
    cbar.ax.tick_params(labelsize=10) 



#------------------ Month Numbering
# Rename Months to Numbers
def MonthRenameToNum(data):
    data.rename(
      columns={'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6','Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'},inplace=True)
    
def MonthRenameToShort(data):
    data.rename(
      columns={1 : 'Jan',2 : 'Feb',3 : 'Mar',4 : 'Apr',5 : 'May',6 : 'Jun',7 : 'Jul',8 : 'Aug',9 : 'Sep',10 : 'Oct',11: 'Nov',12 : 'Dec',},inplace=True)

def MonthRenameStringNumToShort(data):  
    data.rename(columns={'1' : 'Jan','2' : 'Feb','3' : 'Mar','4' : 'Apr','5' : 'May','6' : 'Jun','7' : 'Jul','8' : 'Aug','9' : 'Sep','10' : 'Oct','11': 'Nov','12' : 'Dec'},inplace=True)

#----------------Custom Sorting a Dataframe-------------------------
def sort_pd(key=None,reverse=False):
    def sorter(series):
        series_list = list(series)
        return [series_list.index(i) 
           for i in sorted(series_list,key=key,reverse=reverse)]
    return sorter

def monthsort(df,idx_lvl):
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sort_by_month = sort_pd(key=month_order.index)
    df=df.iloc[sort_by_month(df.index.get_level_values(idx_lvl))]
    return df

#-----
# Change months to numbers
def map_level(df, dct, level):
    dct = {'Jan':'1', 'Feb':'2', 'Mar':'3', 'Apr':'4', 'May':'5', 'Jun':'6', 'Jul':'7',
        'Aug': '8','Sep':'9', 'Oct':'10','Nov':'11','Dec':'12'}
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i==level else names
                      for i, names in enumerate(index.levels)], inplace=True)

#-----
import calendar
month_mapper = dict((v,k) for k,v in enumerate(calendar.month_abbr))


#------------------ BarPlot  
def BarPlotDatavEstimate(data, estimate ,Title):
    
    grdc_runoff = pd.concat([data, estimate], axis=1)
    
    params = {'xtick.labelsize': 8}
    plt.rcParams.update(params) 

    fig = plt.figure(figsize=(12,6))
    fig.set_size_inches(6,4) 
    ax = grdc_runoff.plot(kind = 'bar', logy = False, width=0.8)
    ax.legend( ('Data ', 'Estimate'), bbox_to_anchor=(2, 1) )
    ax.set_ylabel('Annual Runoff [km$^3$])', fontsize=10)
    ax.set_title(Title, fontsize=12)

    ax.grid(True)
    plt.gcf().subplots_adjust(bottom=0.29)
    #plt.savefig('Figures/runoff.pdf')
    params = {'xtick.labelsize': 12}
    plt.rcParams.update(params) 

    rects = ax.patches

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height+1, str('%.1f' % float(height)), ha='center', va='bottom', fontsize=10)

    plt.show()
    
    
 #------------------ To Data Frame without Nulls 
def ToDF(var):
    var=var.to_series()
    var=var[~var.isnull()]
    var = pd.DataFrame(var)
    return var

 #------------------ removleLast4 
def removeLast4(d3):
    d3.columns.names=['','crop']
    d3=d3.stack()
    d3.reset_index(level=['crop'], inplace=True)
    d3['crop']
    d3['crop'] = d3['crop'].map(lambda x: str(x)[:-4])
    return d3


 #------------------ Mann-Kendall Test
from scipy.stats import norm, mstats


def mk_test(x, alpha = 0.05):  
    """   
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics 

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05) 
    """
    n = len(x)

    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-st.norm.cdf(abs(z))) # two tail test
    h = abs(z) > st.norm.ppf(1-alpha/2) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    return trend, h, p, z



 #------------------ Stationarity Test
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(center=False,window=12).mean()
    
    rolstd = timeseries.rolling(center=False,window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)




#------------------- Excel well data grouped by month
def WellToMonthly(WL_data_indexed, date_start, date_end):
    WL_sel=  WL_data_indexed.loc[date_start:date_end]
    
    df = WL_sel
    data = pd.pivot_table(df, values='WLevel(mbgl)', index='datetime', columns=['Station','LAT','LONG','Elevation'])
    d2 = data.groupby(data.index.month).mean()
    
    # Reset index
    Jan = d2.iloc[0].to_frame().reset_index()
    May = d2.iloc[1].to_frame().reset_index()
    Aug = d2.iloc[2].to_frame().reset_index()
    Nov = d2.iloc[3].to_frame().reset_index()
    
    #Add heads
    Jan['Head']=Jan['Elevation']-Jan[1]
    May['Head']=May['Elevation']-May[5]
    Aug['Head']=Aug['Elevation']-Aug[8]
    Nov['Head']=Nov['Elevation']-Nov[11]
    
    # If negative, set to 0
    df=Jan
    num = df._get_numeric_data()
    num[num < 0] = 0
    
    df=May
    num = df._get_numeric_data()
    num[num < 0] = 0
    
    df=Aug
    num = df._get_numeric_data()
    num[num < 0] = 0
    
    df=Nov
    num = df._get_numeric_data()
    num[num < 0] = 0
    
    #Remove NAN's
    df = Jan
    df.dropna(axis=0, how='any',inplace=True)
    
    df = May
    df.dropna(axis=0, how='any',inplace=True)
    
    df = Aug
    df.dropna(axis=0, how='any',inplace=True)
    
    df = Nov
    df.dropna(axis=0, how='any',inplace=True)
    
    return Jan,May,Aug,Nov
    

#------------------- Make Contour Plots
def MakeContourPlot(df,title,cbarmin,cbarmax,savefig_flag=0, plottype=1, nlevels=20 ):
    
    fig = plt.figure(figsize=(10,10))
    ax  = fig.add_axes([0., 0., 1., 1., ])
    ax.axis('tight')
    m = MakeBasemap(ax)
    river = m.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/KrishnaRiver', 'KrishnaRiver', drawbounds = True, color='blue', linewidth=0.5)
    
    import matplotlib.mlab as ml
    contour_df = df
    # Mesh the data for plotting
    # transform to numpy arrays
    x = np.array(contour_df.LAT)
    y = np.array(contour_df.LONG)
    z = np.array(contour_df.Head)
    
    #Generate a regular grid to interpolate the data.
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    
    X, Y = np.meshgrid(xi, yi)
    
    #Interpolate
    Z_gw = ml.griddata(x, y, z, xi, yi, interp='linear')
    
    #Contourshapes Filled or lines
    if plottype==1:
        CS = ax.contourf(Y,X,Z_gw)
    else:
        data=Z_gw
        levels = np.arange(np.min(data[~np.isnan(data)]), np.max(data[~np.isnan(data)]), nlevels)
        CS = ax.contour(Y,X,data, levels, origin='lower',linewidths=1)

    #Well points
    points = m.scatter(df.LONG,df.LAT,c=df.Head,cmap='jet',s=70)

#Colorbar
    cbar = plt.colorbar(CS,shrink=0.7)
    cbar.ax.set_ylabel('Head', size=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_clim(cbarmin, cbarmax)
    
    plt.title(title, size=28)
    if savefig_flag==1:
        plt.savefig('/Users/Anjuli/Desktop/GWImages/'+ title+'.png')
    
    plt.show()
    
    return X,Y,Z_gw


#---------------- Make Interpolation Plots
def MakeInterpolationPlot(data, latlongs):
    from scipy.interpolate import griddata
    
    df=data.reset_index()
    
    # Data Points
    #data = np.array( df[['LAT','LONG','Head']] )
    
    # From data
    points = np.array( df[ ['LAT','LONG'] ])
    values = np.array( df['Head'])
    
    # from Fishnet
    gridx=latlongs.LAT_Center.unique()
    gridy=latlongs.LONG_Center.unique()
    
    grid_x, grid_y = np.mgrid[13.25:19.25+0.5:0.5, 73.75:80.75+0.5:0.5]
    
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_axes([0., 0., 1., 1., ])
    m = MakeBasemap(ax)
    
    im = m.imshow(grid_z1, interpolation='nearest') # Linear_interp
    
    #ax.scatter(points.T[1],points.T[1],c=values,cmap='jet',s=5)
    ax.scatter(df.LONG,df.LAT,c=df.Head,cmap='jet',s=70)
    
    cb =m.colorbar(im,"right", size="5%", pad='2%')
    plt.show()
    
    Linear_Interp=grid_z1
    
    return Linear_Interp


# ---------------Check Normality
def CheckNormality(z):
    from scipy.stats import norm
    import scipy.stats as stats
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    
    ax1 = plt.subplot(121)
    hrange = (0,max(z))
    mu, std = norm.fit(z)
    
    ahist=ax1.hist(z, bins=10, normed=True, alpha=0.6, color='c',range=hrange)
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    ax1.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    #th=plt.title(title)
    th=ax1.set_title(title)
    xh=plt.xlabel('Values')
    yh=plt.ylabel('Density')
    xl=plt.xlim(0,max(z))
    yl=plt.ylim(0,max(ahist[0]+0.01))
    
    
    ax2 = plt.subplot(122)
    qqdata = stats.probplot(z, dist="norm",plot=plt,fit=False)
    title='QQ Plot'
    th=ax2.set_title(title)
    xh=plt.xlabel('Standard Normal Quantiles')
    yh=plt.ylabel('Values')
    fig=plt.gcf()
    fig.set_size_inches(8,4)
    
    plt.show()


# ------------- Plot Semivariogram
def PlotSemivariogram(data, lag_size, num_lags):
    P = np.array( data[['LAT','LONG','Head']] )
    
    # bandwidth
    bw =lag_size # degrees
    
    # lags in increments
    lags = np.arange(0,num_lags,bw)
    
    sv = SV( P, lags, bw )
    
    #sp = cvmodel( P, model=model, hs=lags, bw=bw )
    
    plt.plot( sv[0], sv[1], '.' )
    #plt.plot( sv[0], sv[1], '.-' )
    
    #plot( sv[0], sp( sv[0] ) )
    
    plt.xlabel('Lag [dist]')
    plt.ylabel('Semivariance')
    plt.title('Sample Semivariogram') ;
    #savefig('sample_semivariogram.png',fmt='png',dpi=200)
    plt.show()

#-----------------Plot Kriging

def PlotKriging(station_data, krig_data, title ):
    
    data = station_data.reset_index()
    
    fig = plt.figure(figsize=(10,10))
    ax  = fig.add_axes([0., 0., 1., 1., ])
    m = MakeBasemap(ax)
    
    im = m.imshow(krig_data, interpolation='nearest')
    
    cb =m.colorbar(im,"right", size="5%", pad='2%')
    
    ax.scatter(data.LONG,data.LAT,c=data.Head,cmap='jet',s=70)
    
    cbar.ax.set_ylabel('Kriging Value', size=20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.title('PyKrige '+ title, size=28)
    if savefig_flag==1:
        plt.savefig('/Users/Anjuli/Desktop/GWImages/OrdKrig.png')
    
    plt.show()
    return


#-------- Semivariogram
# SEMIVARIOGRAM
#http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
from scipy.spatial.distance import pdist, squareform

def SVh( P, h, bw ):
    '''
        Experimental semivariogram for a single lag
        (P)      ndarray, data
        pdist(X) - Pairwise distance computes the Euclidean distance between pairs of objects in m-by-n data matrix X
        '''
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    return np.sum( Z ) / ( 2.0 * len( Z ) )


def SV( P, hs, bw ):
    '''
        Experimental variogram for a collection of lags
        (hs)     distances
        '''
    sv = list()
    for h in hs:
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T


# -------- Variogram
# Variograms Models


def C( P, h, bw ):
    '''
        Calculate the sill
        '''
    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )



def opt( fct, x, y, C0, parameterRange=None, meshSize=1000 ):
    if parameterRange == None:
        parameterRange = [ x[1], x[-1] ]
    mse = np.zeros( meshSize )
    a = np.linspace( parameterRange[0], parameterRange[1], meshSize )
    for i in range( meshSize ):
        mse[i] = np.mean( ( y - fct( x, a[i], C0 ) )**2.0 )
    return a[ mse.argmin() ]



def spherical( h, a, C0 ):
    '''
        Spherical model of the semivariogram
        '''
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        if h <= a:
            return C0*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    # if h is an iterable
    else:
        # calcualte the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        #return map( spherical, h, a, C0 )
        return list(map( spherical, h, a, C0 ))


def cvmodel( P, model, hs, bw ):
    '''
        Input:  (P)      ndarray, data
        (model)  modeling function
        - spherical
        - exponential
        - gaussian
        (hs)     distances
        (bw)     bandwidth
        Output: (covfct) function modeling the covariance
        '''
    # calculate the semivariogram
    sv = SV( P, hs, bw )
    # calculate the sill
    C0 = C( P, hs[0], bw )
    # calculate the optimal parameters
    param = opt( model, sv[0], sv[1], C0 )
    # return a covariance function
    covfct = lambda h, a=param: C0 - model( h, a, C0 )
    return covfct




def krige( P, model, hs, bw, u, N ):
    '''
        Input  (P)     ndarray, data
        (model) modeling function
        - spherical
        - exponential
        - gaussian
        (hs)    kriging distances
        (bw)    kriging bandwidth
        (u)     unsampled point
        (N)     number of neighboring
        points to consider
        '''
    
    # covariance function
    covfct = cvmodel( P, model, hs, bw )
    # mean of the variable
    mu = np.mean( P[:,2] )
    
    # distance between u and each data point in P
    d = np.sqrt( ( P[:,0]-u[0] )**2.0 + ( P[:,1]-u[1] )**2.0 )
    # add these distances to P
    P = np.vstack(( P.T, d )).T
    # sort P by these distances
    # take the first N of them
    P = P[d.argsort()[:N]]
    
    # apply the covariance model to the distances
    k = covfct( P[:,3] )
    # cast as a matrix
    k = np.matrix( k ).T
    
    # form a matrix of distances between existing data points
    K = squareform( pdist( P[:,:2] ) )
    # apply the covariance model to these distances
    K = covfct( K.ravel() )
    # re-cast as a NumPy array -- thanks M.L.
    K = np.array( K )
    # reshape into an array
    K = K.reshape(N,N)
    # cast as a matrix
    K = np.matrix( K )
    
    # calculate the kriging weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )
    
    # calculate the residuals
    residuals = P[:,2] - mu
    
    # calculate the estimation
    estimation = np.dot( weights.T, residuals ) + mu
    
    return float( estimation )


# ----Kriging
# Kriging

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

# Variogram must have amplitude and direction
def SK(x,y,v,variogram,grid):
    cov_angulos = np.zeros((x.shape[0],x.shape[0]))
    cov_distancias = np.zeros((x.shape[0],x.shape[0]))
    K = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]-1):
        cov_angulos[i,i:]=np.arctan2((y[i:]-y[i]),(x[i:]-x[i]))
        cov_distancias[i,i:]=np.sqrt((x[i:]-x[i])**2+(y[i:]-y[i])**2)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if cov_distancias[i,j]!=0:
                amp=np.sqrt((variogram[1]*np.cos(cov_angulos[i,j]))**2+(variogram[0]*np.sin(cov_angulos[i,j]))**2)
                K[i,j]=v[:].var()*(1-np.e**(-3*cov_distancias[i,j]/amp))
    K = K + K.T

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distancias = np.sqrt((i-x[:])**2+(j-y[:])**2)
            angulos = np.arctan2(i-y[:],j-x[:])
            amplitudes = np.sqrt((variogram[1]*np.cos(angulos[:]))**2+(variogram[0]*np.sin(angulos[:]))**2)
            M = v[:].var()*(1-np.e**(-3*distancias[:]/amplitudes[:]))
            W = LA.solve(K,M)
            grid[i,j] = np.sum(W*(v[:]-v[:].mean()))+v[:].mean()
    return grid



def exponential_variogram_model(m, d):
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d/(range_/3.))) + nugget



def OK(x,y,v,variogram,grid):
    cov_angulos = np.zeros((x.shape[0],x.shape[0]))
    cov_distancias = np.zeros((x.shape[0],x.shape[0]))
    K = np.zeros((x.shape[0]+1,x.shape[0]+1))
    for i in range(x.shape[0]-1):
        cov_angulos[i,i:]=np.arctan2((y[i:]-y[i]),(x[i:]-x[i]))
        cov_distancias[i,i:]=np.sqrt((x[i:]-x[i])**2+(y[i:]-y[i])**2)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if cov_distancias[i,j]!=0:
                amp=np.sqrt((variogram[1]*np.cos(cov_angulos[i,j]))**2+(variogram[0]*np.sin(cov_angulos[i,j]))**2)
                K[i,j]=v[:].var()*(1-np.e**(-3*cov_distancias[i,j]/amp))
    K = K + K.T
    K[-1,:] = 1
    K[:,-1] = 1
    K[-1,-1] = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            distancias = np.sqrt((i-x[:])**2+(j-y[:])**2)
            angulos = np.arctan2(i-y[:],j-x[:])
            amplitudes = np.sqrt((variogram[1]*np.cos(angulos[:]))**2+(variogram[0]*np.sin(angulos[:]))**2)
            M = np.ones((x.shape[0]+1,1))
            M[:-1,0] = v[:].var()*(1-np.e**(-3*distancias[:]/amplitudes[:]))
            W = LA.solve(K,M)
            grid[i,j] = np.sum(W[:-1,0]*(v[:]))

    return grid



#-------- Intersections of lines

# Function for finding intersections
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return 0






#-------- Functions

# set the colormap and centre the colorbar
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """
        Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
        
        e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
        """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

#--------
# Data frame Joins
def join_dfs (ldf, rdf):
    return ldf.join(rdf, how='inner')

#--------
# Plot Comparisons
def ComparePlot(Data,Modeled, month,cmax, val1, val2):
        fig = plt.figure(figsize=(15,15))
        
        cmap = plt.get_cmap('rainbow')
        cmap.set_under('white')  # Color for values less than vmin
        eps = 0.0; # Very small float such that 0.0 != 0 + eps
        
        
        # MODEL
        Data = np.ma.masked_where(Data == 0, Data)
        
        ax1 = fig.add_subplot(231)
        im1 = ax1.imshow(Data, interpolation='none',vmin=0,vmax = cmax, cmap=cmap)
        #im1 = ax1.imshow(Data, interpolation='none')
        ax1.set_title("Data in month:"+ str(month+1))
        cb1 =fig.colorbar(im1,ax=ax1, shrink=0.7)
        cb1.set_clim(0,cmax)
        
        # DATA
        Modeled = np.ma.masked_where(Modeled == 0, Modeled)
        
        ax2 = fig.add_subplot(232)
        im2 = ax2.imshow(Modeled, interpolation='none',vmin=0,vmax = cmax, cmap=cmap)
        #im2 = ax2.imshow(Modeled, interpolation='none')
        ax2.set_title("Modeled in month:"+ str(month+1))
        cb2 = fig.colorbar(im2, ax=ax2, shrink=0.7)
        cb2.set_clim(0,cmax)
        
        # DIFF
        
        diff = Modeled-Data
        
        elev_min = np.min(diff)
        elev_max = np.max(diff)
        mid_val = 0;
        
        ax3 = fig.add_subplot(233)
        #im3 = ax3.imshow(diff, interpolation='none', cmap=plt.get_cmap('RdBu_r'), clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max))
        im3 = ax3.imshow(diff, interpolation='none',cmap=plt.get_cmap('RdBu_r'), clim = (val1, val2), norm=MidpointNormalize(midpoint=0, vmin=val1, vmax=val2)  )
        ax3.set_title("Difference (Mod-Data) in month:"+ str(month+1))
        cb3 =fig.colorbar(im3, ax=ax3, shrink=0.7)
        
        
        plt.show()

#--------
# With Basemap
#ax1 = fig.add_subplot(231)
#m = MakeBasemap(ax1)
#Data = np.ma.masked_where(Data == 0, Data)
#im1 = m.imshow(Data, interpolation='none')
#ax1.set_title("Data in month:"+ str(month+1))
#cmap = plt.cm.jet
#cmap.set_bad(color='white')
#cb =ax1.colorbar(im1,"right", ax=ax1, size="5%", pad='2%')
#cb.set_clim(0, 2500)

#--------
# Create Dataframes
def CompartmentTS(data):
    df = pd.DataFrame(monthsort(data[:,:,:,'ir'].groupby(level=['time']).sum(),0))
    df.columns=['ir']
    df = df.assign( rf = pd.DataFrame(monthsort(data[:,:,:,'rf'].groupby(level=['time']).sum(),0)) )
    df = df.assign( nc = pd.DataFrame(monthsort(data[:,:,:,'nc'].groupby(level=['time']).sum(),0)) )
    return df


def SubcropCodeToCropName(test):
    test=test.reset_index()
    test['time']=test['time'].astype('category')
    test['crop']=test['subcrop'].map(lambda x: str(x)[:-4])
    test['crop'].replace(MIRCA2000_cropclassesStr2, inplace=True)
    test
    
    # Ensuring the order of months is preserved!!
    sorter = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    test.time.cat.set_categories(sorter, inplace=True)
    return test


def monthly_subcrop_maps (cmap, cropnum, clut, vmax):
    for month in range(0,12):
        #ir = 0;
        #rf =1;
        
        if len(cropnum)==3:
            A1 = runval.CA_Data[:,:,month,clut[0],cropnum[0]].data
            B1 = runval.CA_Data[:,:,month,clut[0],cropnum[1]].data
            C1 = runval.CA_Data[:,:,month,clut[0],cropnum[2]].data
            
            A2 = runval.CA_Data[:,:,month,clut[1],cropnum[0]].data
            B2 = runval.CA_Data[:,:,month,clut[1],cropnum[1]].data
            C2 = runval.CA_Data[:,:,month,clut[1],cropnum[2]].data
            
            irr_subcrop_area = np.nansum(np.dstack((A1,B1,C1)),2)
            rf_subcrop_area = np.nansum(np.dstack((A2,B2,C2)),2)
        
        if len(cropnum)==2:
            A1 = runval.CA_Data[:,:,month,clut[0],cropnum[0]].data
            B1 = runval.CA_Data[:,:,month,clut[0],cropnum[1]].data
            
            A2 = runval.CA_Data[:,:,month,clut[1],cropnum[0]].data
            B2 = runval.CA_Data[:,:,month,clut[1],cropnum[1]].data
            
            irr_subcrop_area = np.nansum(np.dstack((A1,B1)),2)
            rf_subcrop_area = np.nansum(np.dstack((A2,B2)),2)
        
        if len(cropnum)==1:
            irr_subcrop_area = runval.CA_Data[:,:,month,clut[0],cropnum[0]].data
            rf_subcrop_area = runval.CA_Data[:,:,month,clut[1],cropnum[0]].data
        
        
        irr_subcrop_fraction = irr_subcrop_area/pixarea
        rf_subcrop_fraction = rf_subcrop_area/pixarea
        
        # mask nans and mask 0's
        irr_subcrop_frac = np.ma.masked_where(np.isnan(irr_subcrop_fraction.data),irr_subcrop_fraction )
        irr_masked_array = np.ma.masked_where(irr_subcrop_frac  == 0, irr_subcrop_frac )
        
        # mask nans and mask 0's
        rf_subcrop_frac = np.ma.masked_where(np.isnan(rf_subcrop_fraction.data),rf_subcrop_fraction )
        rf_masked_array = np.ma.masked_where(rf_subcrop_frac  == 0, rf_subcrop_frac )
        
        cmap = cmap
        cmap.set_bad(color='grey')
        
        
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(irr_masked_array, cmap=cmap, interpolation ='none', vmin=0, vmax=vmax)
        ax1.set_title("Irrigated: "+MIRCA2000_cropclassesStr2[runval.CA_Data.subcrop.data[cropnum[0]][0:2]]+ " in Month: " +str(month+1))
        cb1 =fig.colorbar(im1,ax=ax1, shrink=0.25)
        #cb1.set_clim(0,1)
        
        
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(rf_masked_array, cmap=cmap, interpolation ='none', vmin=0, vmax=vmax)
        ax2.set_title("Rainfed: "+ MIRCA2000_cropclassesStr2[runval.CA_Data.subcrop.data[cropnum[0]][0:2]]+ " in Month: " +str(month+1))
        cb2 =fig.colorbar(im2,ax=ax2, shrink=0.25)
        #cb2.set_clim(0,1)

    plt.show()



#--------
from mpl_toolkits.basemap import Basemap


def MakeBasemap2(ax):
    # Locate a bounding box
    ll_long = 73.4; ll_lat = 12.9; ur_long = 81.1; ur_lat = 19.6
    # Create Basemap
    
    map = Basemap(llcrnrlon=ll_long,llcrnrlat=ll_lat,urcrnrlon=ur_long,urcrnrlat=ur_lat, ax=ax)
    #             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
    
    # Bring In Shapefile with Parallels and Meredians
    shp_info_basin = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/KrishnaBasin', 'KrishnaBasin', drawbounds = True, color ='k', linewidth=1.0)
    #gridshp_info = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/Krishna_Fishnet', 'Krishna_Fishnet', drawbounds = True, color='grey', linewidth=0.5)
    
    parallels = np.arange(13,20,1) # make latitude lines ever 1 degrees
    meridians = np.arange(70,84,1) # make longitude lines every 1 degrees from 95W to 70W
    
    # labels = [left,right,top,bottom]
    #map.drawparallels(parallels,labels=[True,False,False,False],fontsize=10)
    #map.drawmeridians(meridians,labels=[False,False,False,True],fontsize=10)
    
    return map


#--------
def MakeBasemap3(ax):
    # Locate a bounding box
    ll_long = 73.4; ll_lat = 12.9; ur_long = 81.1; ur_lat = 19.6
    # Create Basemap
    
    map = Basemap(llcrnrlon=ll_long,llcrnrlat=ll_lat,urcrnrlon=ur_long,urcrnrlat=ur_lat, ax=ax)
    #             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
    
    # Bring In Shapefile with Parallels and Meredians
    shp_info_basin = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/KrishnaBasin', 'KrishnaBasin', drawbounds = True, color ='k', linewidth=1.0)
    #gridshp_info = map.readshapefile('/Users/Anjuli/Dropbox (MIT)/Research Shares/Krishna_GAMS/Krishna_ArcGIS/Krishna_Fishnet', 'Krishna_Fishnet', drawbounds = True, color='grey', linewidth=0.5)
    
    parallels = np.arange(13,20,1) # make latitude lines ever 1 degrees
    meridians = np.arange(70,84,1) # make longitude lines every 1 degrees from 95W to 70W
    
    # labels = [left,right,top,bottom]
    map.drawparallels(parallels,labels=[True,False,False,False],fontsize=10, linewidth=0.0)
    map.drawmeridians(meridians,labels=[False,False,False,True],fontsize=10, linewidth=0.0)
    
    return map





#-------------- Abline
import matplotlib.pyplot as plt
import numpy as np

def abline(slope, intercept, colorstyle, label):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, colorstyle, label=label)
    plt.legend(prop={'size':20})


from scipy.optimize import curve_fit
def func(x, A): # this is your 'straight line' y=f(x)
    return A*x

#-------------- Summary Stats
# Function to find slopes
def getSummaryStats(df, var, obs, est, summary_stats, note):
    
    df = df.dropna()
    df = df.reset_index()
    
    # Calculate statistics
    x = df[obs].values
    y = df[est].values
    
    popt, pcov = curve_fit(func, x, y)
    slope = popt[0]
    intercept = 0;
    
    summary_stats = summary_stats.append({'Estimated Variable' :  est,
                                         'Statistic': 'fit slope',
                                         'Value': slope,
                                         'Note':note
                                         } , ignore_index=True)
                                         
                                         
    return summary_stats


#-------------- Scatter Plots
# Function for Scatter Plots

def ScatterPlotObsvEst(df, var, obs, est, Title, ax, colorstyle, mark_size):
    import seaborn as sns
    
    #df.shape
    #dom = runval.Domain.to_series()
    #df = df.join(dom).query('Domain == True')
    
    # Map label to color
    color_labels = df.index.get_level_values(var).unique()
    
    num =len(df.index.get_level_values(var).unique())
    
    color_palette = sns.color_palette(None, num)
    color_map = dict(zip(color_labels, color_palette))
    
    
    df = df.dropna()
    df = df.reset_index()
    df['color'] = df[var].apply(lambda x: color_map[x])
    
    
    # Calculate statistics
    x = df[obs].values
    y = df[est].values
    
    #print(len(x))
    
    popt, pcov = curve_fit(func, x, y)
    slope = popt[0]
    intercept = 0;
    
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    # Plot
    scatter = ax.scatter(df[obs], df[est], c=df['color'], s=mark_size)
    
    
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel('Observed', size=14)
    plt.ylabel('Estimate', size=14)
    plt.tick_params(labelsize=14)
    
    plt.title("Scatter Plot Obs vs Est for " + Title, size=16)
    
    
    # produce a legend with the unique colors from the scatter
    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0,0],[0,0], color=color , marker='o', linestyle='') for color in color_map.values()]
    legend = ax.legend(markers, color_map.keys(), numpoints=1, bbox_to_anchor=(-0.1, -0.3), loc='upper left', ncol=4, title=var, fontsize=15)
    #legend = ax.legend(markers, color_map.keys(), numpoints=1, loc='lower right', ncol=4, title=var, prop={'size':15})
    ax.add_artist(legend)
    
    
    abline(slope,intercept, colorstyle, 'Linear Fit:  y='+str(round(slope,2))+"x")
    abline(1,0, 'k--',"1:1")


#-------WATER BALANCE


def plotWaterBalance(runval):

    BasinArea = runval.BasinArea.values
    BasinArea_km2 = BasinArea/1e6
    mm2km = 1e-6;
    m2Tokm2 =1e-6 

    #PRECIPITATION
    dfP = pd.DataFrame(monthsort(runval.P_Modeled.to_series().groupby(level=['time']).sum(),0))

    #RECHARGE
    dfW = pd.DataFrame(monthsort(runval.W_Modeled.to_series().groupby(level=['time']).sum(),0))

    #ET

    dfE = pd.DataFrame(monthsort(runval.ET_Modeled.to_series().groupby(level=['time']).sum(),0))

    ET_vol_km3 = ToDF(runval.ET_byCompartment_inTime)
    ET_vol_km3 = ET_vol_km3.reset_index()

    ET_vol_km3['ET_mm'] = ET_vol_km3['ET_byCompartment_inTime']/(BasinArea*m2Tokm2)*1e6
    ET_vol_km3

    IRET = ET_vol_km3[ET_vol_km3['lut']=='ir']
    RFET = ET_vol_km3[ET_vol_km3['lut']=='rf']
    NCET = ET_vol_km3[ET_vol_km3['lut']=='nc']

    IRET = IRET.rename(columns={'ET_byCompartment_inTime':'IRET'})
    IRET2 = IRET.set_index('time')['IRET']
    IRET2 = pd.DataFrame(IRET2)

    RFET = RFET.rename(columns={'ET_byCompartment_inTime':'RFET'})
    RFET2 = RFET.set_index('time')['RFET']
    RFET2 = pd.DataFrame(RFET2)

    NCET = NCET.rename(columns={'ET_byCompartment_inTime':'NCET'})
    NCET2 = NCET.set_index('time')['NCET']
    NCET2 = pd.DataFrame(NCET2)

    #SOILMOISTURE
    dfS = pd.DataFrame(monthsort(runval.dS_Modeled.to_series().groupby(level=['time']).sum(),0))

    #PUMPING
    dfQp = pd.DataFrame(monthsort(runval.Qp_Modeled.to_series().groupby(level=['time']).sum(),0))
    #SURFACE DIVERSION
    dfQsd = pd.DataFrame(monthsort(runval.Qsd_Modeled.to_series().groupby(level=['time']).sum(),0))

    #RUNOFF
    #dfR = pd.DataFrame(monthsort(runval.R_Modeled[5,14].to_series().groupby(level=['time']).sum(),0))

    df = ToDF(runval.R_Modeled)
    df
    df = df.reset_index()
    df2 = df[df['r']=='06']
    df3 = df2[df2['c']=='14']
    df3.groupby('time').sum()

    dfR = df3
    dfR = dfR[['time','R_Modeled']]
    dfR = dfR.set_index('time')


    # Merge all dataframes into one, check signs for sources and sinks
    dfs_list = [dfP, -dfE, -dfW, -dfS, dfQp, dfQsd, -dfR] # W is negative when out; sources are pos; sinks negative
    dfs_list2 = [dfP, -IRET2,-RFET2,-NCET2, -dfW, -dfS, dfQp, dfQsd, -dfR]

    df_wb = reduce(join_dfs, dfs_list2) 
    df_wb.head()

    # MODELED
    df_mod = df_wb[['P_Modeled','IRET','RFET','NCET','W_Modeled','dS_Modeled','R_Modeled',
                    'Qp_Modeled','Qsd_Modeled']]
    df_mod = df_mod.rename(columns={'P_Modeled':'Precipitation', 'W_Modeled':'Recharge',
                                    'dS_Modeled':'dS','R_Modeled':'Basin Runoff', 'Qp_Modeled':'Pumping', 
                                    'Qsd_Modeled':'Surface Diversion',
                                    'IRET': 'Irrigated ET',
                                    'RFET': 'Rainfed ET',
                                    'NCET': 'Noncrop ET'
                                   })
    
   


    # Plot---
    plt.figure(figsize=(15,8))
    G = gridspec.GridSpec(1,1)
    axes_1 = plt.subplot(G[0, 0])
    colorpal=['steelblue', 'lightskyblue','seagreen','khaki', 'grey', 'brown','orange','purple','pink'];

    df_mod.plot(kind='bar', stacked='True', ax=axes_1, width=0.8, color=colorpal)
    plt.legend(fontsize=13, loc='upper left', ncol=2, frameon=False)
    plt.ylim([-40,40])
    #plt.title('Monthly Water Balance', fontsize=14)
    plt.ylabel('Volume (km3)', fontsize=14)
    plt.xlabel('Months',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    
    return df_mod

#---------
def plotSumBasinET(runval, casename, savepath):
    
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    ET = pd.DataFrame(runval.ET_Modeled.to_series())
    ET_Data = pd.DataFrame(runval.ET_Data.to_series())
    ET_Rom = pd.DataFrame(runval.ET_ROM_pixel.to_series())

    X_list = [ET_Data, ET, ET_Rom] 
    X_join=reduce(join_dfs, X_list)
    
    df=X_join

    # Correlation
    corr = monthsort(df.groupby('time').sum(),0).corr().iloc[0,1]

    # Plot ############################################
    plt.figure(figsize=(10, 5))
    G = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(G[0, 0])
    monthsort(df.groupby('time').sum(),0).plot(figsize=(10,5), linewidth=5, color=['red','blue','green'], fontsize=20, ax=ax1)
    #plt.title("Sum Basin ET")
    ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(corr,2)), rotation=0, wrap=True, fontsize=15)
    plt.ylabel('Volume (km3)', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('')
    
    plt.legend(['MODIS ET','Estimated ET','Surrogate Model ET'], fontsize=14)
    
    plt.savefig(savepath+'et_timeseries.png', format='png', dpi=1200)
    print('saved in '+savepath) 
    
    plt.show()
    
    col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
    et_summary_stats_slope  = pd.DataFrame(columns = col_names)

    col_names =  ['Case', 'Estimated Variable', 'Statistic','Value', 'Note']
    et_summary_stats_corr = pd.DataFrame(columns = col_names)
    
    '''
    # Correlation of difference
    diff_corr = monthsort(df.groupby('time').sum(),0).diff().corr().iloc[0,1]
    
     # Plot ############################################
    plt.figure(figsize=(10, 5))
    G = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(G[0, 0])   
    monthsort(df.groupby('time').sum(),0).diff().plot(figsize=(10,5), linewidth=5,color=['red','blue','green'], fontsize=20, ax=ax1, title="Month to Month Diff of Sum Basin ET")      
    ax1.text(ax1.get_xlim()[0], ax1.get_ylim()[1]+1, "R2: "+str(round(diff_corr,2)), ha='left', rotation=0, wrap=True, fontsize=15) 
    plt.show()
    '''   
    

    
    # Store summary stats
    et_summary_stats_corr = et_summary_stats_corr.append({#'Case' : casename , 
                         'Estimated Variable' :  "Basin ET",
                         'Statistic': 'correlation',
                         'Value': corr,
                         'Note':'(vs. data and rom)'                
                         } , ignore_index=True)
    '''

    et_summary_stats_corr =et_summary_stats_corr.append({#'Case' : casename , 
                         'Estimated Variable' :  'Basin ET',
                         'Statistic': 'correlation of diff',
                         'Value': diff_corr,
                         'Note':'(vs. data and rom)'                
                         } , ignore_index=True)
    '''

    return et_summary_stats_corr





def ETbyCrop2(runval,CA_dat):  

    m2Tokm2 =1e-6 
    dom = runval.Domain.to_series()
    
    indata=runval;
    BasinArea = indata.BasinArea.values
    BasinArea_km2 = BasinArea/1e6
    
    # Cropland Modeled
    ira= monthsort(ToDF(runval.CA_time[:,:,:,0,:]).groupby(level=['time']).sum(),0) #thousand m2
    rfa= monthsort(ToDF(runval.CA_time[:,:,:,1,:]).groupby(level=['time']).sum(),0)
    ira.columns=['ira']
    rfa.columns=['rfa']
    CA_mod = ira.join(rfa)


    #Parameter ET_byCompartment_inTime(m,l);
    #ET_byCompartment_inTime(m,l) = sum(n$Domain(n), (sum(sc, ETf.l(n,m,l,sc)$(sl(l,sc) and season(sc,m)) *mmTokm *A(n)*m2Tokm2 ))

    ET_vol_km3 = ToDF(runval.ET_byCompartment_inTime)
    ET_vol_km3 = ET_vol_km3.reset_index()

    ET_vol_km3['ET_mm'] = ET_vol_km3['ET_byCompartment_inTime']/(BasinArea*m2Tokm2)*1e6
    ET_vol_km3

    IRET = ET_vol_km3[ET_vol_km3['lut']=='ir']
    RFET = ET_vol_km3[ET_vol_km3['lut']=='rf']
    NCET = ET_vol_km3[ET_vol_km3['lut']=='nc']


    return IRET, RFET, NCET

