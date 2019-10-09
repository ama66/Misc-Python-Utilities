#This is the number of rows for full dataset 1+int((df.index.max() - df.index.min()).total_seconds()/60)
 
#Which you can compare to len(df)
 
#(2) 
#This is the full set of timestamps from start to end of a dataframe
 
#timeindexes= pd.DatetimeIndex(start=dfw.index.min(), end=dfw.index.max(),freq='min')
 
#if every timestamp in the full dataset is in df or the other way around. The expression below should sum to the length of the dataframe
 
#df.index.isin(timeindexes).sum()

## you could rount timestamps to nearest minute for example df.index[3].round("min")
