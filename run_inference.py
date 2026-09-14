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
LOOKBACK = 4  # Looks at 4 steps to predict the next 1

# =====================================================================
# STEP 1: Preprocessing & Vector Transformations
# =====================================================================
def preprocess_cyclone_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # ⚠️ PLACEHOLDER: Ensuring Vmax exists (replace with true values later)
    if 'Vmax' not in df.columns:
        df['Vmax'] = 0.0

    # Ensure chronological order
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
    df['elapsed_hours'] = df.groupby('ID')['datetime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600.0)
    
    scaler_time = MinMaxScaler()
    df['scaled_time'] = scaler_time.fit_transform(df[['elapsed_hours']])

    return df

# =====================================================================
# STEP 2: Window Generation & Track Alignment
# =====================================================================
def prepare_inference_matrices(df, lookback=4):
    feature_cols = ['scaled_time', 'Vmax', 'x', 'y', 'z']
    
    X_list = []
    current_pos_list = []
    target_timestamps = []

    x_idx = feature_cols.index('x')
    y_idx = feature_cols.index('y')
    z_idx = feature_cols.index('z')

    for _, group in df.groupby('ID'):
        group = group.sort_values(by='time')
        feats = group[feature_cols].values
        timestamps = group['time'].values

        if len(feats) < lookback:
            continue

        # Adjust tracking loops to catch the future predicted time points
        for i in range(len(feats) - lookback + 1):
            window = feats[i : i + lookback]
            X_list.append(window)
            
            # Extract absolute 3D position at the final step (t) of the lookback window
            current_pos = window[-1, [x_idx, y_idx, z_idx]]
            current_pos_list.append(current_pos)
            
            # If there is a known true row at t+1, grab its timestamp label
            if i + lookback < len(timestamps):
                target_timestamps.append(int(timestamps[i + lookback]))
            else:
                # Extrapolate next +3 hour timestamp if it's the final row
                last_dt = pd.to_datetime(str(timestamps[-1]), format='%Y%m%d%H')
                next_dt = last_dt + pd.Timedelta(hours=3)
                target_timestamps.append(int(next_dt.strftime('%Y%m%d%H')))

    return np.array(X_list), np.array(current_pos_list), target_timestamps

# =====================================================================
# STEP 3: Mathematical Vector Reconstruction Post-Processing
# =====================================================================
def decode_deltas_to_latlon(predictions_delta_xyz, current_absolute_xyz, timestamps):
    # Reconstruct true absolute 3D coordinates (t + 3 hours)
    future_absolute_xyz = current_absolute_xyz + predictions_delta_xyz
    
    x = future_absolute_xyz[:, 0]
    y = future_absolute_xyz[:, 1]
    z = np.clip(future_absolute_xyz[:, 2], -1.0, 1.0) 

    # Reverse 3D Cartesian coordinates to standard geographic degrees
    pred_lat = np.degrees(np.arcsin(z))
    pred_lon = np.degrees(np.arctan2(y, x))

    # Normalize back into original 0-360 range
    pred_lon = np.where(pred_lon < 0, pred_lon + 360, pred_lon)

    # Format explicitly as a list of tracking records mapped by target timestamp
    output_records = []
    for idx, ts in enumerate(timestamps):
        output_records.append({
            "target_timestamp": ts,
            "predicted_lat": float(round(pred_lat[idx], 4)),
            "predicted_lon": float(round(pred_lon[idx], 4))
        })

    return output_records

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Process local CSV containing the 8 rows
    df_processed = preprocess_cyclone_data(CSV_INPUT_FILE)
    
    # 2. Extract sequences (Yields 5 sliding window steps out of 8 total rows)
    X_val, absolute_positions, target_times = prepare_inference_matrices(df_processed, lookback=LOOKBACK)

    # 3. Predict from local Keras model asset
    model = tf.keras.models.load_model(MODEL_FILENAME)
    predicted_deltas = model.predict(X_val)

    # 4. Convert back to coordinates array
    predictions_list = decode_deltas_to_latlon(predicted_deltas, absolute_positions, target_times)

    # 5. Structure final output envelope as JSON
    final_json_output = json.dumps({"predictions": predictions_list}, indent=4)
    
    # Print clean JSON out to console
    print(final_json_output)

    # Save tracking payload directly to a JSON file
    with open("cyclone_prediction_output.json", "w") as json_file:
        json_file.write(final_json_output)
    print("\nResults successfully exported to 'predictions_output.json'")
