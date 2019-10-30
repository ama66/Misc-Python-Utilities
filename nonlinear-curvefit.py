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
    

## polynomial fitting in sklearn 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
plt.style.use('fivethirtyeight')

    
file_path="/Users/Ammar/flights.xlsx"


Json_Path="/Users/Ammar/flight1.json"


df=pd.read_excel(file_path, sheet_name="Flight1")


print(df.head) 

df.columns=[str(i).lower() for i in df.columns]
            
assert df.apply(lambda x: x.is_monotonic).all() , "check monotonicity!" 


Coeffs={}
for i in cols:
    range=[]
    Dic_Coeff={"2nd order":None , "4th order":None , "3rd order":None}
    x=df[i].values[0:4]
    y=df['ppm'].values[0:4]
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    
    range.append(df[i].values[3])
    range.append(df[i].values[9])
    
    Dic_Coeff["Range"]=range 
    
    ####################
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    Dic_Coeff['2nd order']=[model.intercept_[0] , model.coef_[0][1], model.coef_[0][2] ]

    ypred=model.intercept_[0] + model.coef_[0][1] * x + model.coef_[0][2] *x*x 
    
    #### transforming the data to include another axis
    
    x=df[i].values[3:10]
    y=df['ppm'].values[3:10]
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=4)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)

    ypred=model.intercept_[0] + model.coef_[0][1] * x + model.coef_[0][2] *x*x + model.coef_[0][3]*x*x*x + model.coef_[0][4]*x*x*x*x 
    
    
    np.max(ypred,axis=0)
    np.max(ypred,axis=0)

    Dic_Coeff['4th order']=[model.intercept_[0],model.coef_[0][1] , model.coef_[0][2] , model.coef_[0][3] , model.coef_[0][4] ]

    # transforming the data to include another axis
    x=df[i].values[9:]
    y=df['ppm'].values[9:]
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_poly_pred = model.predict(x_poly)
    
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)

    
    ypred=model.intercept_[0] + model.coef_[0][1] * x + model.coef_[0][2] *x*x + model.coef_[0][3] *x*x*x
    np.max(ypred,axis=0)

    
    Dic_Coeff['3rd order']=[model.intercept_[0],model.coef_[0][1] , model.coef_[0][2] , model.coef_[0][3] ]
    
    Coeffs[i]=Dic_Coeff


with open(Json_Path, 'w') as fp:
    json.dump(Coeffs, fp , indent="\t")



Coeffs.keys()

def eval_fit(z ,dic):
    z1,z2=dic['Range']
    if  0 <= z <=  z1 :
        a0,a1,a2 =dic['2nd order']
        return a0+a1*z+a2*z**2
    elif z1 < z <= z2:
         a0,a1,a2,a3,a4 =dic['4th order']
         return a0+a1*z+a2*z**2+a3*z**3+a4*z**4
    else:
        a0,a1,a2,a3 =dic['3rd order']
        return a0+a1*z+a2*z**2+a3*z**3
            
