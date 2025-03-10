import geopandas as gpd

SHAPEFILE_PATH = r"C:\Jerson\SUNASS\2025\3_Marzo\giz_serv\Cami_Yaku\geo_data\centroides_georeferencial.shp"

try:
    gdf = gpd.read_file(SHAPEFILE_PATH)
    print("Shapefile cargado exitosamente:")
    print(gdf.head())
except Exception as e:
    print(f"Error al cargar el shapefile: {e}")