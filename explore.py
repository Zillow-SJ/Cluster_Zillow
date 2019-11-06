import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns
df = prep_df()
sns.pairplot(df)
