import pandas as pd
import numpy as np
import statsmodels.api as sm
#import statsmodels.api as sm
#df = pd.read_stata("./BES2015_W9_v3.0-2.dta")
df = pd.read_stata("./BES2015_W8_v4.0-1.dta")
pd.set_option('display.max_columns', None)
for col_name in df.columns:
    print(col_name)

#print(df.groupby("monarch")["monarch"].count())

#refactoring EU Referendum Vote
print(df["euRefVote"])
df.drop(df[df['euRefVote'] == "I would/will not vote"].index, inplace = True)
df.drop(df[df['euRefVote'] == "Don't know"].index, inplace = True)
df["euRefVote"] = df["euRefVote"].map({"Stay/remain in the EU":0,"Leave the EU":1})

#refactoring euUndermineIdentity
df.drop(df[df['euUndermineIdentity'] == "Don't know"].index, inplace = True)
df["euUndermineIdentity"] = df["euUndermineIdentity"].map({"Agree":1,"Strongly agree":2,"Neither agree nor disagree":0,"Disagree":-1,"Strongly disagree":-2})

#refactoring monarch figure
df.drop(df[df['monarch'] == "Don't know"].index, inplace = True)
df["monarch"] = df["monarch"].map({"Agree":1,"Strongly agree":2,"Neither agree nor disagree":0,"Disagree":-1,"Strongly disagree":-2})


#refactoring Auth-Liberal Scale
df["al_scale"] = df["al_scale"].replace({"Libertarian":0,"Authoritarian":10})



new_df = df[["monarch","al_scale","euUndermineIdentity","euRefVote"]]

new_df = new_df.dropna()

#dependent variable
x= new_df[["al_scale","euUndermineIdentity","euRefVote"]]
x= sm.add_constant(x)
y= new_df["monarch"]
model = sm.OLS(y,x).fit()
print(model.summary())
#Modeling