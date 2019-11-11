### Running file of pipeline planning, process, exploration, and delivery

- Built SQL query to bring in all columns from properties_2017, predictions_2017,          architecturelstyletype, propertylandusetype, and typeconstructiontype

- Dropped columns with over 20% NaN values

- Dropped columns with high correlation to each other and others providing no value        (parcel id, id, columns with only one number, etc)

- Dropped bathroomcnt column due to high correlation with bathrooms column.
- Dropped bedroomcnt column due to high correlation with bedrooms column. 
- Dropped calculartedbathnbr column due to high correlation with bathrooms column. 
- Dropped finishedsquarefeet12 column due to high correlation with sqft colunm. 
- Dropped calculatedfinishedsquarefeet column due to high correlation with sqft column. 
- Dropped censustractandblock column due to high correlation with fips column.
- Dropped propertylanduseandblock column due to high correlation with fips column.
- Dropped regionidcity Nan rows that were missing; it was determined the effect of         dropping rows had less potential for negitive impact than to impute said row. 
- Dropped roomcnt due to large amount of unresolvable nan's.
- Dropped landtaxvaluedollarcnt column due to high correlation with tax_value column. 
- Dropped taxamount column due to high correlation with tax_value column.


- Dropped assessment year as it is only 2016 or 2017

- Dropped prperty use codes that were strings, lost 106 rows.

- Running linear regression on cleaned data

- added IQR outlier function to prep file

- Scatterplot function to compare variables


- Ran t test for tax value by quartile. 

- Binned train dataset by logerror

- Plotted stdev and mean of tax value

- Ran Silouette Score, outcomes:
wit(n) = with number of variables
#score tax/log = 0.122
#score bedrooms/log = 0.738
#score bathrooms/log = 0.508
#score yearbuilt/log = 0.448
#score sqft/log = -0.37
#score structuretaxvaluedollarcnt/log = 0.1212
#score structuretax/price_sqft/lotsize = 0.145 / 0.754(wit4) !!!!!!!!!!!!!!!!!!!!
#score price_sqft/lotsize = 0.0156 / 0.140(wit3)
#score structuretax/lotsize = 0.755 / 0.848(wit3)
#score structruetax/price_sqft = 0.116 / 0.232(wit3)



- plotted stdev and mean of tax value

- Attempted to use classifier on y cluster. Proved fruitless because clusters were too out of balance.

- Tax_per_sqft was created to represent the price per square foot. 