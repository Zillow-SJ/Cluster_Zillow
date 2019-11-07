### Running file of pipeline planning, process, exploration, and delivery

- Built SQL query to bring in all columns from properties_2017, predictions_2017, architecturelstyletype, propertylandusetype, and typeconstructiontype

- Dropped columns with over 20% NaN values

- Dropped columns with high correlation to each other and others providing no value (parcel id, id, columns with only one number, etc)

- Dropped bathroomcnt column due to high correlation with bathrooms column.
- Dropped bedroomcnt column due to high correlation with bedrooms column. 
- Dropped calculartedbathnbr column due to high correlation with bathrooms column. 
- Dropped finishedsquarefeet12 column due to high correlation with sqft colunm. 
- Dropped calculatedfinishedsquarefeet column due to high correlation with sqft column. 
- Dropped censustractandblock column due to high correlation with fips column.
- Dropped propertylanduseandblock column due to high correlation with fips column.
- Dropped regionidcity Nan rows that were missing; it was determined the effect of          dropping rows had less potential for negitive impact than to impute said row. 
- Dropped roomcnt due to large amount of unresolvable nan's.
- Dropped landtaxvaluedollarcnt column due to high correlation with tax_value column. 
- Dropped taxamount column due to high correlation with tax_value column.


- Dropped assessment year as it is only 2016 or 2017

- Dropped prperty use codes that were strings, lost 106 rows.