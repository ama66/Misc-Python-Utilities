import matplotlib.pyplot as plt
%matplotlib inline 
import warnings
import numpy as np
import pandas as pd
from lmfit import Minimizer, Parameters, report_fit
plt.style.use('seaborn')
from lmfit.models import ExponentialModel, ConstantModel 
from lmfit import CompositeModel, Model

file_path="/Users/Ammar/Flightickets.xlsx"
df=pd.read_excel(file_path)

def crv_fit(x,y):
    
    def residual(pars, x , data=None):
        epsilon=1e-6
        y1=pars['A1']*np.exp(x/pars['tau1'])
        y2=pars['A2']*np.exp(x/pars['tau2'])
        y3=pars['c']
        model = y1+y2+y3 
       # model= list(map(lambda x: x if x > 0 else 1.0 , y1+y2+y3))
        
       # model=np.log10(list(map(lambda x: x if x > 0 else epsilon, y1+y2+y3)))
        if data is None:
            return model 
        return (model - data)


    
    pfit = Parameters()
    pfit.add(name='A1', value=3300)
    pfit.add(name='deltA',vary=True, value=1500)
    pfit.add(name='A2', value=1800, expr='A1-deltA')
    pfit.add(name='tau1', value=340, min=1)
    pfit.add(name='dtau', value=800, min=1,vary=True)
    pfit.add(name='tau2', expr='tau1+dtau' )
    pfit.add(name='c', value=-15)

    myfit = Minimizer(residual, pfit,
                      fcn_args=(x,), fcn_kws={'data': y})

    result = myfit.leastsq()

    best_fit= result.residual + y


    report_fit(result.params)

    plt.plot(x,y, 'bo')

    plt.plot(x,best_fit, 'ro', label='best fit')
    plt.legend(loc='best')
    plt.show();
    print("Error in Y")
    print(best_fit - y)
    
    
y=df['annual'].values
#y[0]=1.0
cols=[ i  for i in df.columns if i!="ant"]
for col in cols:
    x=df[col].values
    crv_fit(x,y)
    
    
