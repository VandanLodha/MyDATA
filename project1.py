import pandas as pd   #for data frame 
import numpy as np   #for numerical calculation
import matplotlib.pyplot as plt    #for visualization
import statsmodels.formula.api as smf

#load the data of oil and gas set 
data =  pd.read_csv(r'C:\Users\Vandan\Desktop\IIT ROORKEE\Project\Oil and Gas 1932-2014.csv')

#to know the column list properly
data.columns

#dropping the unnecessary columns form the data as to predict oil we need only oil data set
data.drop([  'id', 'gas_prod55_14', 'gas_price_2000_mboe',
       'gas_price_2000', 'gas_price_nom', 'gas_value_nom', 'gas_value_2000',
       'gas_value_2014', 'gas_exports', 'net_gas_exports_bcf',
       'net_gas_exports_mboe', 'net_gas_exports_value',
       'net_gas_exports_valuePOP','iso3numeric',  'eiacty',
       'population', 'pop_maddison', 'sovereign', 'mult_nom_2000',
       'mult_nom_2014', 'mult_2000_2014'],inplace=True,axis=1)

#now again looking the data colummns
data.columns

#looking out the correlation of oil data with the oil_price_nom.
data.oil_price_2000.corr(data.oil_prod32_14)  # 0.04054413267351227
data.oil_price_2000.corr(data.oil_price_nom)   # 0.8976705044623309
data.oil_price_2000.corr(data.oil_value_nom)    #0.211447086455295
data.oil_price_2000.corr(data.oil_value_2000)   #0.18383053058483748
data.oil_price_2000.corr(data.oil_value_2014)   #0.18383053058483745
data.oil_price_2000.corr(data.oil_gas_value_nom)  #0.19812508833887324
data.oil_price_2000.corr(data.oil_gas_value_2000)   #0.16620370801124681
data.oil_price_2000.corr(data.oil_gas_value_2014)   #0.1662037080112468
data.oil_price_2000.corr(data.oil_gas_valuePOP_nom)  #0.1664498411767312
data.oil_price_2000.corr(data.oil_gas_valuePOP_2000)   #0.10017695103123628
data.oil_price_2000.corr(data.oil_gas_valuePOP_2014)   #0.10017695103123625
data.oil_price_2000.corr(data.oil_exports)    # 0.02596712179949656
data.oil_price_2000.corr(data.net_oil_exports)  #-0.012966746569062689
data.oil_price_2000.corr(data.net_oil_exports_mt)   #-0.012966746569062701
data.oil_price_2000.corr(data.net_oil_exports_value)   #-0.021641659229807427
data.oil_price_2000.corr(data.net_oil_exports_valuePOP)  #0.06225548518301419
data.oil_price_2000.corr(data.net_oil_gas_exports_valuePOP)   #0.06347541351306372
                         
                         
                         
#dropping the unwanted columns from the data.
data.drop(['cty_name', 'oil_prod32_14','oil_value_nom', 'oil_value_2000', 'oil_value_2014',
       'oil_gas_value_nom', 'oil_gas_value_2000', 'oil_gas_value_2014',
       'oil_gas_valuePOP_nom', 'oil_gas_valuePOP_2000',
       'oil_gas_valuePOP_2014', 'oil_exports', 'net_oil_exports',
       'net_oil_exports_mt', 'net_oil_exports_value',
       'net_oil_exports_valuePOP', 'net_oil_gas_exports_valuePOP'],inplace=True,axis=1)


#remaining data columns 
data.columns

#drop the raw data from the data as the oil_price_nom and oil_price_2000 remains same for every country.
data.drop(data.index[83:15521], inplace=True)

#finding corrrelation of the data
data.corr()

#making model after seeing the corr()
model = smf.ols('oil_price_2000 ~ year + oil_price_nom', data=data).fit()
model.params
model.summary()

model1=smf.ols('oil_price_2000~year',data = data).fit()  
model1.summary()  # r-square = 0.806 but p>t of year is 0.741 which is not acceptable

#plot the influencer plot to know  the raw
import statsmodels.api as sm
sm.graphics.influence_plot(model)  #from the plot identifying the influencer raw and rremoving from the data

#delete the raws from the data
data_new = data.drop(data.index[[82,81,79,80,76]],axis=0)

#write new model with the help of new data set
model_new = smf.ols('oil_price_2000 ~ year + oil_price_nom', data=data_new).fit()
model_new.summary()  # r-square = 0.714 and p>t of year is 0.091 which is acceptable


#after seeing the model_new summary it is clear that the model is acceptable. So predicting the oil price
oil_price_pred = model_new.predict(data_new)
oil_price_resid = oil_price_pred - data_new.oil_price_2000
oilprice_rmse = np.sqrt(np.mean(oil_price_resid*oil_price_resid))  #9.17


#plotting the model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model_new)

#check linearity of the model
plt.scatter(data_new.oil_price_2000,oil_price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#assupmtions about errors
plt.hist(model_new.resid_pearson)  #errors are not normally distrubuted.


import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model_new.resid_pearson, dist="norm", plot=pylab)   

# Residuals VS Fitted Values 
plt.scatter(oil_price_pred,model_new.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")























