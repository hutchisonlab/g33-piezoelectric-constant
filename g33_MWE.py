#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress
from scipy.signal import find_peaks

# Load data
a = pd.pandas.read_csv('data.txt', delimiter='\t')

# Correct baseline
x = a.iloc[30:530]['Time (s)']
y = a.iloc[30:530]['VoltageB (V)']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
a['Corrected Voltage'] = a['VoltageB (V)'] - slope * a['Time (s)']
baseline = np.mean(a.iloc[30:530]['Corrected Voltage'])
a['Corrected Voltage'] = a['Corrected Voltage'] - baseline

# Identify Force and Voltage Peaks
# Average initial force
force_avg = np.mean(a.iloc[30:530]['ForceA (N)'])
force_std = np.std(a.iloc[30:530]['ForceA (N)'])
# Calculate force baseline
force_baseline = force_avg + 3*force_std + 0.2
# Calculate force peaks
force_peaks, force_properties = find_peaks(a['ForceA (N)'],
    height=force_baseline, distance=100, prominence=0)
force_index = a['ForceA (N)'].index[force_peaks]
force_times = a.iloc[force_index]['Time (s)']
force_times = force_times.values
force_values = a['ForceA (N)'].values[force_peaks]
force_values_rel = force_properties['prominences']
# Calculate voltage peaks
voltage_peaks, voltage_properties = \
    find_peaks(a.iloc[530:]['Corrected Voltage'], distance=100, prominence=0)
voltage_index = a.iloc[530:]['Corrected Voltage'].index[voltage_peaks]
voltage_times = a.iloc[voltage_index]['Time (s)']
voltage_times = voltage_times.values
voltage_values = a.iloc[530:]['Corrected Voltage'].values[voltage_peaks]
voltage_values_rel = voltage_properties['prominences']

# Process results
area = 0.035 * 0.035  # meters
separation = 0.00004  # meters
x = force_values_rel / area
y = voltage_values_rel * 1000 / separation
# Calculate robust linear regression
x_rlm = sm.add_constant(x)
huber_t = sm.RLM(y, x_rlm, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
# Calculate slope percent standard error
percent_std_err = hub_results.bse[1] / hub_results.params[1] * 100
slope = hub_results.params[1]
intercept = hub_results.params[0]
print('Response\tIntercept\tSlope-Percent-Standard-Error\t')
print(str(slope) + '\t' + str(intercept) + '\t' + str(percent_std_err))

