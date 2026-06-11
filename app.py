from flask import Flask, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Force headless mode for server environments
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
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

    # Compressing the images inside the ZIP using standard DEFLATE compression
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown_sensor")
            
            dates_raw = sensor.get("dates", [])
            displacement = sensor.get("displacement", [])
            temperature = sensor.get("temperature", [])

            if not dates_raw or not displacement or not temperature:
                continue
            if not (len(dates_raw) == len(displacement) == len(temperature)):
                continue

            dates_dt = [excel_to_datetime(d) for d in dates_raw]

            df = pd.DataFrame({
                'DateTime': dates_dt,
                'Displacement': displacement,
                'Temperature': temperature
            })
            df.dropna(subset=['DateTime'], inplace=True)
            df.set_index('DateTime', inplace=True)

            if df.empty:
                continue

            df_hourly = df.resample('h').mean().dropna()
            if df_hourly.empty:
                continue

            # Calculate mapping to a circle
            coords = df_hourly.index.map(lambda x: get_polar_coords(x))
            df_hourly['X'], df_hourly['Y'] = zip(*coords)

            # --- DYNAMIC THICKNESS SCALING ---
            # Automatically scales the visual width between 1.5 and 10.0 
            # regardless of whether data ranges in thousandths of an inch or full millimeters.
            min_disp = df_hourly['Displacement'].min()
            max_disp = df_hourly['Displacement'].max()
            disp_range = max_disp - min_disp

            if disp_range > 0:
                widths = 1.5 + ((df_hourly['Displacement'].values - min_disp) / disp_range) * 8.5
            else:
                widths = np.full_like(df_hourly['Displacement'].values, 3.0)

            # Temperature color mapping parameters
            min_temp = df_hourly['Temperature'].min()
            max_temp = df_hourly['Temperature'].max()
            norm = Normalize(vmin=min_temp, vmax=max_temp)

            # Create the segments for LineCollection
            points = np.array([df_hourly['X'], df_hourly['Y']]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            seg_widths = widths[:-1]

            # Set up wider figure canvas layout to cleanly fit the side legends
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            
            # Construct line collection and bind scalar array values for proper colorbar mapping
            lc = LineCollection(segments, linewidths=seg_widths, cmap='coolwarm', norm=norm, capstyle='round')
            lc.set_array(df_hourly['Temperature'].values[:-1])
            ax.add_collection(lc)

            # Add month labels around the circle perimeter
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, month in enumerate(months):
                month_angle = np.pi/2 - (i / 12) * 2 * np.pi
                tx, ty = 1.20 * np.cos(month_angle), 1.20 * np.sin(month_angle)
                ax.text(tx, ty, month, ha='center', va='center', fontweight='bold', fontsize=8)

            # Formatting core plot spatial bounds
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-1.35, 1.35)
            ax.set_aspect('equal')
            ax.axis('off')

            # Clean Title Placement 
            ax.text(0, 1.45, f"Sensor: {sensor_id}", ha='center', va='bottom', fontsize=11, fontweight='bold')

            # --- ADD TEMPERATURE COLORBAR ---
            cbar = fig.colorbar(lc, ax=ax, pad=0.08, shrink=0.5, anchor=(0.0, 0.2))
            cbar.set_label('Temperature (°C)', fontsize=8, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)

            # --- ADD DISPLACEMENT SIZE LEGEND ---
            # Creates 4 clean proxy interval lines on the upper right side
            legend_samples = np.linspace(min_disp, max_disp, 4)
            legend_elements = []
            for val in legend_samples:
                w = 1.5 + ((val - min_disp) / disp_range) * 8.5 if disp_range > 0 else 3.0
                legend_elements.append(
                    Line2D([0], [0], color='darkgray', lw=w, label=f"{val:.3f} in")
                )
            
            ax.legend(
                handles=legend_elements,
                title="Displacement",
                loc="upper left",
                bbox_to_anchor=(1.05, 0.9),
                frameon=False,
                title_fontproperties={'weight': 'bold', 'size': 8},
                prop={'size': 8}
            )

            # Save graphic out to memory buffer package
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', transparent=True)
            plt.close(fig)  
            buf.seek(0)

            zf.writestr(f"{sensor_id}.png", buf.read())

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='crack_plots.zip'
    )
