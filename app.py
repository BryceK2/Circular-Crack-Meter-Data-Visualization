from flask import Flask, request, send_file, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile
from datetime import datetime, timedelta

app = Flask(__name__)

# Convert Excel serial date → datetime
def excel_to_datetime(excel_date):
    return datetime(1899, 12, 30) + timedelta(days=float(excel_date))

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json()
    sensors = data.get("sensors", [])

    if not sensors:
        return jsonify({"error": "No sensor data provided"}), 400

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown")

            xplot = np.array(sensor["ew"])
            yplot = np.array(sensor["ns"])
            dates = np.array(sensor["dates"], dtype=float)

            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_title(f"Tilt Meter: {str(sensor_id)}", fontsize=14, fontweight='bold', pad=34)

            # Determine ring spacing dynamically
            base_spacing = 0.01
            max_tilt = max(np.abs(xplot).max(), np.abs(yplot).max())

            # Calculate a scaling factor: ceil(max_tilt / 0.03)
            scale_factor = int(np.ceil(max_tilt / 0.03))
            if scale_factor < 1:
                scale_factor = 1

            spacing = base_spacing * scale_factor
            radii = [spacing * i for i in range(1, 4)]  # 3 rings

            # # Draw circles
            label_offset = spacing * 0.02  # small fraction of spacing
            offset = spacing * 0.5 + 0.005
            for r in radii:
                theta = np.linspace(0, 2*np.pi, 300)
                ax.plot(r*np.cos(theta), r*np.sin(theta), color='black', lw=1)
                ax.text(r + label_offset, -offset, f"{r:.2f}°", va='center', fontsize=10, zorder=4)

            # Draw crosshairs
            ax.plot([-radii[2], radii[2]], [0, 0], color='black', lw=1, zorder=0)
            ax.plot([0, 0], [-radii[2], radii[2]], color='black', lw=1, zorder=0)

            # Convert dates to day-of-year for fixed Jan–Dec coloring
            dates_dt = np.array([excel_to_datetime(d) for d in dates])
            day_of_year = np.array([d.timetuple().tm_yday for d in dates_dt])
            
            sc = ax.scatter(
                xplot, yplot,
                c=day_of_year,
                s=50,
                cmap='jet',
                edgecolors='none',
                zorder=3
            )
            sc.set_clim(1, 365)

             # Colorbar with month ticks
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            cbar = plt.colorbar(sc, ax=ax, pad=0.18)
            cbar.set_ticks(month_starts)
            cbar.set_ticklabels(month_labels)

            # Formatting
            padding = spacing * 0.2
            limit = radii[-1] + padding
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect('equal')

            # Remove box
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # Labels (positioned just outside outer ring + scaled offset)
            label_offset_factor = 2.5
            ax.text(0, radii[-1] + label_offset*label_offset_factor, "North",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.text(0, -radii[-1] - label_offset*label_offset_factor, "South",
                    ha='center', va='top', fontsize=12, fontweight='bold')
            ax.text(radii[-1] + label_offset*label_offset_factor, 0, "East",
                    ha='left', va='center', fontsize=12, fontweight='bold')
            ax.text(-radii[-1] - label_offset*label_offset_factor, 0, "West",
                    ha='right', va='center', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.subplots_adjust(right=0.85, bottom=0.15, top=0.90)

            # Save image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, transparent=True)
            plt.close(fig)
            buf.seek(0)

            zf.writestr(f"{sensor_id}.png", buf.read())

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='tilt_plots.zip'
    )
