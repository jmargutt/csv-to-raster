import pandas as pd
import numpy as np
import math
import rasterio
from rasterio.transform import from_origin
import os
from tqdm import tqdm
import mercantile
from pyquadkey2 import quadkey


def calculate_tile_bbox(row):
    qk = str(quadkey.from_geo((row['latitude'], row['longitude']), 14))
    b = mercantile.bounds(mercantile.quadkey_to_tile(qk))
    step_lon = abs(b.west - b.east)
    step_lat = abs(b.north - b.south)
    return step_lon, step_lat


def meters_to_longitude(meters, latitude):
    """ convert meters to decimal degrees, based on latitude
    World circumference measured around the equator is 40,075.017 km (source: wikipedia)
    40075.017 / 360 = 111.319491667 km/deg
    """
    return meters / (111.319491667 * 1000 * math.cos(latitude * (math.pi / 180)))


def meters_to_latitude(meters):
    """ convert meters to decimal degrees, based on latitude
    World circumference measured around the equator is 40,075.017 km (source: wikipedia)
    40075.017 / 360 = 111.319491667 km/deg
    """
    return meters / (111.132952778 * 1000)


def longitude_to_meters(degrees, latitude):
    """ convert decimal degrees to meters, based on latitude
    1 arc degree at the equator 111.132952778 km
    """
    return math.cos(latitude * (math.pi / 180)) * abs(degrees) * 111.319491667 * 1000


def latitude_to_meters(degrees):
    """ convert decimal degrees to meters, based on latitude
    World circumference measured around the equator is 40,007.863 km (source: wikipedia)
    40,007.863 / 360 = 111.132952778 km/deg
    """
    return abs(degrees) * 111.132952778 * 1000


def csv_to_raster(input_csv, output_raster, resolution=2400):
    """ convert csv file to raster """

    # read csv
    df = pd.read_csv(input_csv)

    # calculate raster cell size
    # step_lon = meters_to_longitude(resolution, df.latitude.mean())
    # step_lat = meters_to_latitude(resolution)
    df['step_lon'], df['step_lat'] = zip(*df.apply(calculate_tile_bbox, axis=0))
    step_lon = df['step_lon'].mean()
    step_lat = df['step_lat'].mean()

    # get minimum latitude and longitude
    lat_min = df.latitude.min()
    lon_min = df.longitude.min()
    # get extent in longitude and latitude
    Dlat = abs(df.latitude.max() - lat_min)
    Dlon = abs(df.longitude.max() - lon_min)
    # convert extent in number of raster cells
    size_lat = round(Dlat / step_lat) + 1
    size_lon = round(Dlon / step_lon) + 1
    # initialize empty array
    arr = np.empty((size_lat, size_lon))
    arr[:] = np.nan

    # loop over the csv file and fill the array
    for ix, row in df.iterrows():
        dlat = abs(row['latitude'] - lat_min)
        dlon = abs(row['longitude'] - lon_min)
        ilat = round((size_lat - 1) * dlat / Dlat)
        ilon = round((size_lon - 1) * dlon / Dlon)
        # correct for rasterio flip
        ilat = size_lat - ilat - 1
        arr[ilat, ilon] = row['rwi']

    # transform array into raster
    transform = from_origin(lon_min - step_lon * 0.5,
                            df.latitude.max() + step_lat * 0.5,
                            step_lon, step_lat)
    new_raster = rasterio.open(output_raster, 'w', driver='GTiff',
                               height=arr.shape[0], width=arr.shape[1],
                               count=1, dtype=str(arr.dtype),
                               crs='EPSG:4326',
                               transform=transform)
    # save raster
    new_raster.write(arr, 1)
    new_raster.close()


def calculate_error(df_in, df_out):
    df_in = pd.read_csv(df_in)
    df_out = pd.read_csv(df_out)
    df_out.columns = ['longitude', 'latitude', 'rwi']
    df_out = df_out.dropna(subset=['rwi']).reset_index(drop=True)
    dist = pd.DataFrame()
    for ix, row in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        lat = row['latitude']
        lon = row['longitude']
        df_out['dist'] = np.sqrt(np.power(df_out['latitude']-lat, 2) + np.power(df_out['longitude']-lon, 2))
        idxmin = df_out['dist'].idxmin()
        min_err_lat = df_out.iloc[idxmin]['latitude'] - lat
        min_err_lat_m = latitude_to_meters(min_err_lat)
        min_err_lon = df_out.iloc[idxmin]['longitude'] - lon
        min_err_lon_m = longitude_to_meters(min_err_lon, lat)

        dist = dist.append(pd.Series({'latitude': lat,
                                      'longitude': lon,
                                      'err_lat_deg': min_err_lat,
                                      'err_lat_met': min_err_lat_m,
                                      'err_lon_deg': min_err_lon,
                                      'err_lon_met': min_err_lon_m
                                      }), ignore_index=True)
    return dist


if __name__ == "__main__":

    input_dir = "relative-wealth-index-april-2021"
    output_dir = input_dir+"-geotiff"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dist_all = pd.DataFrame()
    for file in os.listdir(input_dir):
        print(f"processing {file}")
        input_filepath = os.path.join(input_dir, file)
        output_filepath = os.path.join(output_dir, file.replace('.csv', '.tif'))
        csv_to_raster(input_filepath, output_filepath)

        print(f"calculating errors")
        # execute command gdal_translate
        output_filepath_table = output_filepath.replace('.tif', '.csv')
        os.system(rf'C:\Users\JMargutti\Anaconda3\envs\typhoon\Library\bin\gdal_translate.exe '
                  rf'-of xyz -co ADD_HEADER_LINE=YES -co COLUMN_SEPARATOR="," {output_filepath} {output_filepath_table}')
        dist = calculate_error(input_filepath, output_filepath_table)
        dist_all = dist_all.append(pd.Series({'file': file,
                                              'mean_err_lat_deg': dist['err_lat_deg'].mean(),
                                              'mean_err_lat_met': dist['err_lat_met'].mean(),
                                              'max_err_lat_deg': dist['err_lat_deg'].max(),
                                              'max_err_lat_met': dist['err_lat_met'].max(),
                                              'mean_err_lon_deg': dist['err_lon_deg'].mean(),
                                              'mean_err_lon_met': dist['err_lon_met'].mean(),
                                              'max_err_lon_deg': dist['err_lon_deg'].max(),
                                              'max_err_lon_met': dist['err_lon_met'].max()
                                              }), ignore_index=True)
        print(dist_all.iloc[-1])

    dist_all.to_csv('errors.csv')