import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from RegImpute.StaticImpute import StaticImpute



df = pd.read_csv("test_data/test_missing.csv")
data_cols = df.columns[1:] #define columns to be imputed
data = df[data_cols].values.astype(np.float64) #extract data from those columns

model = Ridge(alpha=0.7, fit_intercept=True, normalize=True) #Choosing to use ridge regression since it's fast and robust.
imp_inst = StaticImpute(data,'static','regressionImpute',model=model,fillmethod="row_median",max_iter=50,mse=1e-9) #specify imputation model
imputed = imp_inst.impute() #run imputation and return results results
df[data_cols] = imputed # overwrite nans with imputed data

df.to_csv("imputed_results.txt",index=False)
	



	














































