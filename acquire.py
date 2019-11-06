import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import env

url = env.get_db_url('zillow')


def wrangle_zillow():
    df = pd.read_sql("""
    SELECT 
    p17.transactiondate,p.id,p.bathroomcnt as bathrooms,p.bedroomcnt as bedrooms, p.calculatedfinishedsquarefeet as sqft, p.taxvaluedollarcnt as tax_value, `architecturalstyledesc`,a.*, p.*,c.*,p17.*
    FROM propertylandusetype pl
    JOIN properties_2017 p ON p.propertylandusetypeid = pl.propertylandusetypeid
    JOIN predictions_2017 p17 ON p17.id = p.id
    LEFT JOIN architecturalstyletype a USING(architecturalstyletypeid)
    LEFT JOIN typeconstructiontype c USING(typeconstructiontypeid)
    WHERE 
    pl.propertylandusedesc LIKE '%%Single Family%%'
    AND 
    p.calculatedfinishedsquarefeet IS NOT NULL
    and
    p.bedroomcnt > 0
    and 
    p.bathroomcnt > 0
    and
    p.taxvaluedollarcnt > 0
    and
    p.propertycountylandusecode != "010G"
    and
    p.propertycountylandusecode != "010M"

    """
    ,url)
    return df

