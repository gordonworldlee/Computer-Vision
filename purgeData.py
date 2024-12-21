import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import os




def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Set the path
save_path = "/Users/gordonlee/Desktop/VS Code project test/Computer-Vision"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Read data
file_path = "/Users/gordonlee/Desktop/VS Code project test/Computer-Vision/dp_pt_test.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# Extract data and convert datetime to numerical timestamps
dp_pressure = df['DP sensor'].values
pt_pressure = df['PT sensor'].values


dp_time_raw = pd.to_numeric(df['Time (DP sensor)'].values) / 1e9
pt_time_raw = pd.to_numeric(df['Time (PT sensor)'].values) / 1e9

# Normalize time to start at 0
dp_time = dp_time_raw - dp_time_raw[0]
pt_time = pt_time_raw - pt_time_raw[0]

# Calculate sampling frequencies
dp_fs = 1 / np.mean(np.diff(dp_time))
pt_fs = 1 / np.mean(np.diff(pt_time))


# Apply filters to all sensors
# DP Sensor filtering
dp_savgol = savgol_filter(dp_pressure, window_length=51, polyorder=3)
dp_double_savgol = savgol_filter(dp_savgol, window_length=31, polyorder=2)
dp_butter = butter_lowpass_filter(dp_pressure, cutoff=2.0, fs=dp_fs, order=3)

# PT Sensor filtering
pt_savgol = savgol_filter(pt_pressure, window_length=51, polyorder=3)
pt_double_savgol = savgol_filter(pt_savgol, window_length=31, polyorder=2)
pt_butter = butter_lowpass_filter(pt_pressure, cutoff=2.0, fs=pt_fs, order=3)


# Create plots
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 6))

# Plot DP Sensor data
ax1.plot(dp_time, dp_pressure, 'b-', alpha=0.15, label='Raw Data', linewidth=0.5)
ax1.plot(dp_time, dp_savgol, 'r-', label='Savitzky-Golay Filter', linewidth=1.5, alpha=0.7)
ax1.plot(dp_time, dp_double_savgol, 'g-', label='Double Filtered', linewidth=2)
ax1.plot(dp_time, dp_butter, 'k-', label='Butterworth Filter', linewidth=2, alpha=0.7)

# Set y-axis limit for dp_pressure graph
ax1.set_ylim(-0.001, 0.3)
ax1.set_title('DP Sensor Data')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pressure (DP sensor)')
ax1.legend()

# Add more x-axis ticks for DP sensor
dp_time_max = np.ceil(dp_time.max())
xticks_dp = np.linspace(0, dp_time_max, 20)  # 20 tick marks
ax1.set_xticks(xticks_dp)
ax1.tick_params(axis='both', labelsize=10)


# Plot PT Sensor data
ax2.plot(pt_time, pt_pressure, 'b-', alpha=0.15, label='Raw Data', linewidth=0.5)
ax2.plot(pt_time, pt_savgol, 'r-', label='Savitzky-Golay Filter', linewidth=1.5, alpha=0.7)
ax2.plot(pt_time, pt_double_savgol, 'g-', label='Double Filtered', linewidth=2)
ax2.plot(pt_time, pt_butter, 'k-', label='Butterworth Filter', linewidth=2, alpha=0.7)



ax2.set_ylim(14, 270)
ax2.set_title('PT Sensor: Time vs Pressure', fontsize=14)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Pressure', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(fontsize=10)

# Add more x-axis ticks for PT sensor
pt_time_max = np.ceil(pt_time.max())
xticks_pt = np.linspace(0, pt_time_max, 20)  # 20 tick marks
ax2.set_xticks(xticks_pt)
ax2.tick_params(axis='both', labelsize=10)

plt.tight_layout()

# Save the plot
plot_path = os.path.join(save_path, 'Purge_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# Save the filtered data
results_df = pd.DataFrame({
    'DP_Time': dp_time,
    'DP_Raw': dp_pressure,
    'DP_Savitzky_Golay': dp_savgol,
    'DP_Double_Filtered': dp_double_savgol,
    'DP_Butterworth': dp_butter,
    'PT_Time': pt_time,
    'PT_Raw': pt_pressure,
    'PT_Savitzky_Golay': pt_savgol,
    'PT_Double_Filtered': pt_double_savgol,
    'PT_Butterworth': pt_butter,

})

data_path = os.path.join(save_path, 'Purge_analysis.xlsx')
results_df.to_excel(data_path, index=False)
print(f"Data saved to: {data_path}")


plt.show()