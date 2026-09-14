import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_FILENAME = "cyclone_path_predictor.keras"
CSV_INPUT_FILE = "cyclone_data.csv"
LOOKBACK = 8          
FORECAST_STEPS = 3    

# =====================================================================
# STEP 1: Preprocessing & Vector Transformations
# =====================================================================
def preprocess_cyclone_data(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'Vmax' not in df.columns:
        df['Vmax'] = 0.0

    df = df.sort_values(by=['ID', 'time']).reset_index(drop=True)

    # Convert coordinates to 3D Cartesian projection system
    df['lon_adjusted'] = np.where(df['lon'] > 180, df['lon'] - 360, df['lon'])
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon_adjusted'])

    df['x'] = np.cos(lat_rad) * np.cos(lon_rad)
    df['y'] = np.cos(lat_rad) * np.sin(lon_rad)
    df['z'] = np.sin(lat_rad)
    df = df.drop(columns=['lon_adjusted'])

    # Parse temporal data & Scale tracking lifecycle hours
    df['datetime'] = pd.to_datetime(df['time'].astype(str), format='%Y%m%d%H')
    df['elapsed_hours'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds() / 3600.0
    
    scaler_time = MinMaxScaler()
    df['scaled_time'] = scaler_time.fit_transform(df[['elapsed_hours']])

    return df, scaler_time

# =====================================================================
# STEP 2: Mathematical Coordinate Conversions
# =====================================================================
def xyz_to_latlon(x, y, z):
    z_clipped = np.clip(z, -1.0, 1.0)
    pred_lat = np.degrees(np.arcsin(z_clipped))
    pred_lon = np.degrees(np.arctan2(y, x))
    pred_lon = np.where(pred_lon < 0, pred_lon + 360, pred_lon)
    return float(round(pred_lat, 4)), float(round(pred_lon, 4))

def latlon_to_xyz(lat, lon):
    lon_adj = lon - 360 if lon > 180 else lon
    lat_r = np.radians(lat)
    lon_r = np.radians(lon_adj)
    return np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)

# =====================================================================
# STEP 3: RECURSIVE MULTI-STEP FORECASTING LOOP
# =====================================================================
def run_recursive_forecast(df, model, scaler_time, lookback=8, steps=3):
    feature_cols = ['scaled_time', 'Vmax', 'x', 'y', 'z']
    
    x_idx = feature_cols.index('x')
    y_idx = feature_cols.index('y')
    z_idx = feature_cols.index('z')
    vmax_idx = feature_cols.index('Vmax')

    if len(df) < lookback:
        raise ValueError(f"Dataset contains {len(df)} rows, but lookback requires {lookback}.")

    current_window = df.tail(lookback)[feature_cols].values.copy()
    
    last_row = df.iloc[-1]
    current_time_str = str(int(last_row['time']))
    last_dt = pd.to_datetime(current_time_str, format='%Y%m%d%H')
    current_elapsed_hours = last_row['elapsed_hours']
    current_vmax = last_row['Vmax'] 

    output_records = []

    for step in range(1, steps + 1):
        input_matrix = np.expand_dims(current_window, axis=0)
        predicted_delta = model.predict(input_matrix, verbose=0)[0] # Extract array dimension index

        current_absolute_xyz = current_window[-1, [x_idx, y_idx, z_idx]]
        future_absolute_xyz = current_absolute_xyz + predicted_delta

        pred_lat, pred_lon = xyz_to_latlon(future_absolute_xyz[0], future_absolute_xyz[1], future_absolute_xyz[2])
        norm_x, norm_y, norm_z = latlon_to_xyz(pred_lat, pred_lon)

        last_dt += pd.Timedelta(hours=3)
        target_timestamp = int(last_dt.strftime('%Y%m%d%H'))
        current_elapsed_hours += 3.0
        
        scaled_time_val = scaler_time.transform([[current_elapsed_hours]])[0][0]

        output_records.append({
            "forecast_horizon": f"+{step * 3} hours",
            "target_timestamp": target_timestamp,
            "predicted_lat": pred_lat,
            "predicted_lon": pred_lon
        })

        new_row = np.zeros(len(feature_cols))
        new_row[feature_cols.index('scaled_time')] = scaled_time_val
        new_row[vmax_idx] = current_vmax
        new_row[x_idx] = norm_x
        new_row[y_idx] = norm_y
        new_row[z_idx] = norm_z

        current_window = np.vstack([current_window[1:], new_row])

    return output_records

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    df_processed, time_scaler = preprocess_cyclone_data(CSV_INPUT_FILE)
    model = tf.keras.models.load_model(MODEL_FILENAME)
    
    predictions_list = run_recursive_forecast(
        df_processed, model, time_scaler, lookback=LOOKBACK, steps=FORECAST_STEPS
    )

    final_json_output = json.dumps({"predictions": predictions_list}, indent=4)
    print(final_json_output)

    with open("cyclone_prediction_output.json", "w") as json_file:
        json_file.write(final_json_output)
