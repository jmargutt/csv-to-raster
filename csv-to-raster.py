import pandas as pd
import numpy as np
import math
import rasterio
from rasterio.transform import from_origin
import os


def meters_to_decimal_degrees(meters, latitude):
    """ convert meters to decimal degrees, based on latitude
    Measured around the poles, world circumference is 40,007.863 km (source: wikipedia)
    40007.863 / 360 = 111.132952778 m/deg
    """
    return meters / (111.132952778 * 1000 * math.cos(latitude * (math.pi / 180)))


def csv_to_raster(input_csv, output_raster, resolution=2400):
    """ convert csv file to raster """

    # read csv
    df = pd.read_csv(input_csv)

    # calculate raster cell size
    step = meters_to_decimal_degrees(resolution, df.latitude.mean())
    # get minimum latitude and longitude
    lat_min = df.latitude.min()
    lon_min = df.longitude.min()
    # get extent in longitude and latitude
    Dlat = abs(df.latitude.max() - lat_min)
    Dlon = abs(df.longitude.max() - lon_min)
    # convert extent in number of raster cells
    size_lat = round(Dlat / step) + 1
    size_lon = round(Dlon / step) + 1
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
    transform = from_origin(lon_min - step * 0.5,
                            df.latitude.max() + step * 0.5,
                            step, step)
    new_raster = rasterio.open(output_raster, 'w', driver='GTiff',
                               height=arr.shape[0], width=arr.shape[1],
                               count=1, dtype=str(arr.dtype),
                               crs='EPSG:4326',
                               transform=transform)
    # save raster
    new_raster.write(arr, 1)
    new_raster.close()


if __name__ == "__main__":

    input_dir = "test"
    output_dir = input_dir+"-geotiff"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        print(f"processing {file}")
        input_filepath = os.path.join(input_dir, file)
        output_filepath = os.path.join(output_dir, file.replace('.csv', '.tif'))
        csv_to_raster(input_filepath, output_filepath)