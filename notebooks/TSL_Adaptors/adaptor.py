from TSL_Adaptors.mlcore_adaptor import read_data_mlcore
from TSL_Adaptors.zinc_adaptor import read_data_zinc2
from TSL_Adaptors.cldc_adaptor import read_data_cldc
from TSL_Adaptors.pallet_plat_adaptor import read_data_pallet_plant


def get_monitoring_tables(
    dbutils,
    spark,
    catalog_details,
    db_name,
    table_details=None,
):
    model_name = table_details.get("model_name", "").lower()
    if "zinc" in model_name:
        return read_data_zinc2(dbutils, spark, table_details)
    if "cldc" in model_name:
        return read_data_cldc(dbutils, spark, table_details)
    if "pallet" in model_name or "pellet" in model_name:
        return read_data_pallet_plant(dbutils, spark, table_details)
    else:
        return read_data_mlcore(dbutils, spark,catalog_details,db_name,table_details)
