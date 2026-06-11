from flask import Flask, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Force headless mode for server environments
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime, timedelta

app = Flask(__name__)

def excel_to_datetime(excel_date):
    """Converts an Excel serial date float from Apps Script to a Python datetime."""
    try:
        return datetime(1899, 12, 30) + timedelta(days=float(excel_date))
    except Exception:
        return None

def get_polar_coords(dt, radius=1.0):
    """Maps a datetime to (x, y) on a circle where Jan 1 0:00 is at 12:00."""
    total_seconds_in_year = 365.25 * 24 * 3600
    start_of_year = pd.Timestamp(year=dt.year, month=1, day=1)
    seconds_elapsed = (dt - start_of_year).total_seconds()
    
    # In polar coordinates, 12:00 is pi/2. 
    # Clockwise means we subtract the angle as time increases.
    angle = np.pi/2 - (seconds_elapsed / total_seconds_in_year) * 2 * np.pi
    return radius * np.cos(angle), radius * np.sin(angle)

@app.route('/', methods=['POST'])
def process_plots():
    data = request.get_json()
    if not data or "sensors" not in data:
        return jsonify({"error": "Missing payload or 'sensors' array"}), 400

    sensors = data.get("sensors", [])
    if not sensors:
        return jsonify({"error": "No sensor data provided"}), 400

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown_sensor")
            
            dates_raw = sensor.get("dates", [])
            displacement = sensor.get("displacement", [])
            temperature = sensor.get("temperature", [])

            # Skip processing if any critical arrays are empty or mismatched
            if not dates_raw or not displacement or not temperature:
                continue
            if not (len(dates_raw) == len(displacement) == len(temperature)):
                continue

            # Convert Apps Script serial dates into datetimes
            dates_dt = [excel_to_datetime(d) for d in dates_raw]

            # 1. Load the data into DataFrame
            df = pd.DataFrame({
                'DateTime': dates_dt,
                'Displacement': displacement,
                'Temperature': temperature
            })
            df.dropna(subset=['DateTime'], inplace=True)
            df.set_index('DateTime', inplace=True)

            if df.empty:
                continue

            # 2. Downsample to hourly readings
            df_hourly = df.resample('H').mean().dropna()
            if df_hourly.empty:
                continue

            # 3. Calculate mapping to a circle (Clockwise from 12:00)
            coords = df_hourly.index.map(lambda x: get_polar_coords(x))
            df_hourly['X'], df_hourly['Y'] = zip(*coords)

            # 4. Define visual attributes (Thickness and Color)
            base_width = 1.0
            scale_factor = 20.0 
            widths = base_width + scale_factor * df_hourly['Displacement'].values

            # Temperature color mapping (Blue to Red using 'coolwarm')
            norm = Normalize(vmin=df_hourly['Temperature'].min(), vmax=df_hourly['Temperature'].max())
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(norm(df_hourly['Temperature'].values))

            # 5. Create the segments for LineCollection
            points = np.array([df_hourly['X'], df_hourly['Y']]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Shift attributes to segments
            seg_widths = widths[:-1]
            seg_colors = colors[:-1]

            # 6. Plotting
            fig, ax = plt.subplots(figsize=(10, 10))
            lc = LineCollection(segments, linewidths=seg_widths, colors=seg_colors, capstyle='round')
            ax.add_collection(lc)

            # Add month labels around the circle
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, month in enumerate(months):
                month_angle = np.pi/2 - (i / 12) * 2 * np.pi
                tx, ty = 1.18 * np.cos(month_angle), 1.18 * np.sin(month_angle)
                ax.text(tx, ty, month, ha='center', va='center', fontweight='bold', fontsize=10)

            # Formatting the plot
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.title(f"Sensor {sensor_id}\nCrack Displacement (Width) and Temperature (Color)", pad=20)
            
            # Save visual image directly to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
            plt.close(fig)  # Clear current figure memory
            buf.seek(0)

            # Add file to ZIP payload named after the sensor ID
            zf.writestr(f"{sensor_id}.png", buf.read())

    # Prepare complete zip payload for response return
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='crack_plots.zip'
    )

if __name__ == '__main__':
    # Default port set to 5000, Render can bind to environment specific variables automatically
    app.run(debug=True, host='0.0.0.0', port=5000)
