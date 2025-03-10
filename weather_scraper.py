import os
import logging
import sys
import time
import random
from datetime import datetime
import numpy as np

import pandas as pd
import geopandas as gpd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from tqdm import tqdm



# =============================================================================
# CONFIGURATION
# =============================================================================
SHAPEFILE_PATH = r"geo_data\centroides_georeferencial.shp"
OUTPUT_DIR = "data\\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eps_daily_weather_pipe_single_batch.csv")

START_DATE = "2023-02-01"
END_DATE   = "2024-01-30"

# List of HOURLY weather variables we need (no "_spread" columns)
HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm", "boundary_layer_height", "wet_bulb_temperature_2m",
    "total_column_integrated_water_vapour", "is_day", "sunshine_duration", "albedo",
    "snow_depth_water_equivalent", "shortwave_radiation", "direct_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant",
    "direct_normal_irradiance_instant", "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant"
]

# Flood API daily variables
FLOOD_VARIABLES = [
    "river_discharge", "river_discharge_mean", "river_discharge_median",
    "river_discharge_max", "river_discharge_min", "river_discharge_p25",
    "river_discharge_p75"
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("weather_eps_daily_single_batch.log", mode="w")
    ]
)

def main():
    """Fetch entire date-range HOURLY data for all EPS in a single batch (aggregated to daily),
       fetch entire date-range FLOOD data for all EPS, merge them, and also merge geo info."""
    logging.info("=== Starting EPS daily weather script with single batch processing (pipe-delimited) ===")

    # 1) Load and preprocess shapefile
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")
    
    gdf = gpd.read_file(SHAPEFILE_PATH)
    logging.info(f"Loaded {len(gdf)} records from {SHAPEFILE_PATH}.")

    # Standardize column names and ensure 'eps'
    gdf.columns = [c.strip().lower() for c in gdf.columns]
    if 'eps' not in gdf.columns:
        logging.warning("'eps' column missing; using 'UNKNOWN_EPS'.")
        gdf['eps'] = "UNKNOWN_EPS"

    # Extract lat/lon from geometry (assuming points/centroids)
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    valid_mask = (
        gdf["latitude"].notna() &
        gdf["longitude"].notna() &
        gdf["latitude"].between(-90, 90) &
        gdf["longitude"].between(-180, 180)
    )
    gdf = gdf[valid_mask].copy()
    logging.info(f"After lat/lon filter: {len(gdf)} rows remain.")

    # Group by EPS to get a single lat/lon "center" per EPS
    eps_centers = (
        gdf.groupby("eps", as_index=False)
           .agg({"latitude": "mean", "longitude": "mean"})
           .rename(columns={"latitude": "center_lat", "longitude": "center_lon"})
    )
    logging.info(f"Computed {len(eps_centers)} unique EPS centers.")

    # 2) Setup Open-Meteo client with caching & retries
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    # 3) Prepare date range
    start_dt = pd.Timestamp(START_DATE)
    end_dt   = pd.Timestamp(END_DATE)
    logging.info(f"Requesting data from {start_dt.date()} to {end_dt.date()} for all {len(eps_centers)} EPS locations.")

    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    flood_url = "https://flood-api.open-meteo.com/v1/flood"

    # 4) Process all EPS in a single batch
    all_results = []
    
    for _, row in tqdm(eps_centers.iterrows(), total=len(eps_centers), desc="Processing EPS"):
        eps_name = row["eps"]
        lat = row["center_lat"]
        lon = row["center_lon"]
        
        # --- (A) Fetch HOURLY weather in one call ---
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC"
        }
        df_weather_daily = fetch_hourly_as_daily(client, weather_url, weather_params, eps_name)
        if df_weather_daily is None or df_weather_daily.empty:
            logging.warning(f"No weather data for EPS={eps_name}. Skipping flood request.")
            continue

        # --- (B) Fetch FLOOD data in one call ---
        flood_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "daily": FLOOD_VARIABLES,
            "timezone": "UTC"
        }
        df_flood_daily = fetch_flood_daily(client, flood_url, flood_params, eps_name)

        # Convert date_utc columns in both DataFrames to timezone-aware datetime for merging
        df_weather_daily["date_utc"] = pd.to_datetime(df_weather_daily["date_utc"], utc=True)
        if df_flood_daily is not None and not df_flood_daily.empty:
            df_flood_daily["date_utc"] = pd.to_datetime(df_flood_daily["date_utc"], utc=True)
            df_merged = pd.merge(df_weather_daily, df_flood_daily, on=["date_utc", "eps"], how="left")
        else:
            df_merged = df_weather_daily

        all_results.append(df_merged)
        
        # Small delay between requests to avoid rate-limiting
        time.sleep(random.uniform(8, 15))

    # 5) Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        logging.info(f"Final rows retrieved: {len(final_df)}")
    else:
        final_df = pd.DataFrame(columns=["eps","date_utc"] + HOURLY_VARIABLES + FLOOD_VARIABLES)
        logging.warning("No data retrieved for any EPS!")

    # 6) Merge lat/lon (or other geo columns) into final_df
    final_df = pd.merge(final_df, eps_centers, on="eps", how="left")

    # 7) Sort & save final DataFrame
    final_df["date_utc"] = pd.to_datetime(final_df["date_utc"], utc=True)
    final_df = final_df.sort_values(["eps", "date_utc"]).reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        sep='|',
        decimal='.',
        float_format='%.2f'
    )
    logging.info(f"Saved final CSV to: {OUTPUT_FILE}")
    logging.info("=== Pipeline complete ===")


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def fetch_hourly_as_daily(client, base_url, params, eps_name, max_attempts=5):
    """
    Makes one request for HOURLY data over the entire date range,
    handles rate-limit errors (including waiting 100s on hourly limits),
    aggregates to daily means, and returns a DataFrame.
    """
    attempt = 0
    while attempt < max_attempts:
        responses = client.weather_api(base_url, params=params)
        if not responses:
            logging.warning(f"No weather response for EPS={eps_name}, attempt {attempt+1}")
            attempt += 1
            time.sleep(60)
            continue

        resp0 = responses[0]
        if isinstance(resp0, dict) and resp0.get("error", False):
            reason = resp0.get("reason", "")
            if "Minutely API request limit exceeded" in reason:
                logging.error(f"Minutely rate limit hit for EPS={eps_name}: {reason}. Sleeping 60s.")
                attempt += 1
                time.sleep(60)
                continue
            elif "Hourly API request limit exceeded" in reason:
                logging.error(f"Hourly rate limit hit for EPS={eps_name}: {reason}. Sleeping 100s.")
                attempt += 1
                time.sleep(100)
                continue
            else:
                logging.error(f"Error in weather response for EPS={eps_name}: {reason}")
                return None
        break

    if attempt >= max_attempts:
        logging.error(f"Exceeded max attempts for weather data EPS={eps_name}.")
        return None

    hourly_obj = resp0.Hourly()
    if hourly_obj is None:
        logging.warning(f"No HOURLY data object for EPS={eps_name}")
        return None

    epoch_times = hourly_obj.Time()
    if not epoch_times:
        logging.warning(f"No HOURLY timestamps for EPS={eps_name}")
        return None

    dt_start = pd.to_datetime(hourly_obj.Time(), unit="s", utc=True)
    dt_end   = pd.to_datetime(hourly_obj.TimeEnd(), unit="s", utc=True)
    dt_interval = hourly_obj.Interval()

    idx_hourly = pd.date_range(
        start=dt_start,
        end=dt_end,
        freq=pd.Timedelta(seconds=dt_interval),
        inclusive="left"
    )
    df_hourly = pd.DataFrame({"datetime_utc": idx_hourly})
    for i, var in enumerate(HOURLY_VARIABLES):
        df_hourly[var] = hourly_obj.Variables(i).ValuesAsNumpy()
    df_hourly["date"] = df_hourly["datetime_utc"].dt.date
    df_daily = df_hourly.groupby("date", as_index=False).mean(numeric_only=True)
    df_daily.rename(columns={"date": "date_utc"}, inplace=True)
    df_daily.insert(0, "eps", eps_name)
    return df_daily


def fetch_flood_daily(client, base_url, params, eps_name, max_attempts=5):
    """
    Makes one request for FLOOD daily data over the entire date range,
    handles rate-limit errors (including waiting 100s on hourly limits),
    and returns a DataFrame with columns: eps, date_utc, and FLOOD_VARIABLES.
    """
    attempt = 0
    while attempt < max_attempts:
        responses = client.weather_api(base_url, params=params)
        if not responses:
            logging.warning(f"No flood response for EPS={eps_name}, attempt {attempt+1}")
            attempt += 1
            time.sleep(60)
            continue

        resp0 = responses[0]
        if isinstance(resp0, dict) and resp0.get("error", False):
            reason = resp0.get("reason", "")
            if "Minutely API request limit exceeded" in reason:
                logging.error(f"Minutely flood rate limit for EPS={eps_name}: {reason}. Sleeping 60s.")
                attempt += 1
                time.sleep(60)
                continue
            elif "Hourly API request limit exceeded" in reason:
                logging.error(f"Hourly flood rate limit for EPS={eps_name}: {reason}. Sleeping 100s.")
                attempt += 1
                time.sleep(100)
                continue
            else:
                logging.error(f"Error in flood response for EPS={eps_name}: {reason}")
                return None
        break

    if attempt >= max_attempts:
        logging.error(f"Exceeded max attempts for flood data EPS={eps_name}.")
        return None

    daily_obj = resp0.Daily()
    if daily_obj is None:
        logging.warning(f"No FLOOD daily data for EPS={eps_name}")
        return None

    epoch_times = daily_obj.Time()
    if not epoch_times:
        logging.warning(f"No FLOOD daily timestamps for EPS={eps_name}")
        return None

    dt_start = pd.to_datetime(daily_obj.Time(), unit="s", utc=True)
    dt_end   = pd.to_datetime(daily_obj.TimeEnd(), unit="s", utc=True)
    dt_interval = daily_obj.Interval()

    idx_daily = pd.date_range(
        start=dt_start,
        end=dt_end,
        freq=pd.Timedelta(seconds=dt_interval),
        inclusive="left"
    )
    df_flood = pd.DataFrame({"date_utc": idx_daily})
    for i, var in enumerate(FLOOD_VARIABLES):
        df_flood[var] = daily_obj.Variables(i).ValuesAsNumpy()
    df_flood.insert(0, "eps", eps_name)
    return df_flood


if __name__ == "__main__":
    main()