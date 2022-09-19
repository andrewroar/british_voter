import pandas as pd
import numpy as np
import statsmodels.api as sm
#import statsmodels.api as sm
#df = pd.read_stata("./BES2015_W9_v3.0-2.dta")
df = pd.read_stata("./BES2015_W8_v4.0-1.dta")
pd.set_option('display.max_columns', None)
#for col_name in df.columns:
#    print(col_name)



#print(df.groupby("monarch")["monarch"].count())

#refactoring Age to float
df["age"] = df["age"].astype(float)

#refactoring Britishness (scale from 1 to 7)

df.drop(df[df['britishness'] == "Don't know"].index, inplace = True)
df["britishness"] = df["britishness"].replace({"Not at all British":1,"Very strongly British":7})
df["britishness"] = df["britishness"].astype(float)


#refactoring EU Referendum Vote
df.drop(df[df['p_eurefvote'] == "Don't know"].index, inplace = True)
df["p_eurefvote"] = df["p_eurefvote"].map({"I voted to remain":0,"I voted to leave":1})


#refactoring euUndermineIdentity
df.drop(df[df['euUndermineIdentity'] == "Don't know"].index, inplace = True)
df["euUndermineIdentity"] = df["euUndermineIdentity"].map({"Agree":1,"Strongly agree":2,"Neither agree nor disagree":0,"Disagree":-1,"Strongly disagree":-2})

#refactoring monarch figure
df.drop(df[df['monarch'] == "Don't know"].index, inplace = True)
df["monarch"] = df["monarch"].map({"Agree":1,"Strongly agree":2,"Neither agree nor disagree":0,"Disagree":-1,"Strongly disagree":-2})


#refactoring Auth-Liberal Scale
df["al_scale"] = df["al_scale"].replace({"Libertarian":0,"Authoritarian":10})

#Create the new DF with the targeted variables
new_df = df[["ukCitizen","monarch","al_scale","euUndermineIdentity","p_eurefvote","age","britishness"]]
new_df = new_df.dropna()

#dependent variable
x= new_df[["al_scale","euUndermineIdentity","p_eurefvote","age","britishness"]]
x= sm.add_constant(x)
y= new_df["monarch"]


#Modeling
model = sm.OLS(y,x).fit()
print(model.summary())