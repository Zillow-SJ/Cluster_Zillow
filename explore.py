import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns
import pandas_profiling
df = prep.prep_df()
df_2 = df.drop(columns= ["logerror"])
explore_df = pd.Series(df_2.corrwith(df["logerror"]))
explore_df.nlargest(n=5)
explore_df.nsmallest(n=5)
profile = df.profile_report()
rejected_variables = profile.get_rejected_variables(threshold=0.9)
profile