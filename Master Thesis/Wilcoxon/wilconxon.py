import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# --- 1. Load Your Raw Data ---
raw_data = {
    'Q1': [3, 4, 3, 4, 3, 3],
    'Q2': [4, 5, 5, 5, 5, 5],
    'Q3': [3, 4, 3, 4, 4, 4],
    'Q4': [5, 4, 4, 4, 4, 4],
    'Q5': [3, 3, 3, 3, 4, 3],
    'Q6': [5, 5, 5, 4, 5, 5],
    'Q7': [5, 5, 5, 4, 6, 5],
    'Q8': [3, 4, 3, 3, 3, 3],
    'Q9': [4, 4, 3, 4, 4, 4],
    'Q10': [5, 5, 5, 5, 6, 6],
    'Q11': [5, 4, 4, 4, 4, 4],
    'Q12': [4, 4, 4, 5, 6, 4],
}
df_raw = pd.DataFrame(raw_data)

# Define the neutral point for your Likert scale
neutral_point = 4.0

# --- 2. Perform Calculations and Wilcoxon Test for Each Question ---
results_data = [] # To store results for our table

for q_name in df_raw.columns:
    n = len(df_raw[q_name]) # Number of participants
    median_score = df_raw[q_name].median() # Median of raw scores
    q1 = df_raw[q_name].quantile(0.25) # 25th percentile
    q3 = df_raw[q_name].quantile(0.75) # 75th percentile
    iqr = f"{q1:.2f} - {q3:.2f}" # Formatted IQR string

    # Calculate differences from the neutral point for Wilcoxon test
    diffs = df_raw[q_name] - neutral_point

    # Perform Wilcoxon Signed-Rank Test
    # The test needs at least one non-zero difference.
    if len(diffs[diffs != 0]) > 0:
        # 'alternative='two-sided'' checks if median is higher OR lower than neutral.
        # 'method='exact'' is suitable for small sample sizes.
        stat, p_value = wilcoxon(diffs, alternative='two-sided', method='exact')
    else:
        # If all responses are the neutral point, or no variation from it,
        # it's not significantly different.
        stat = np.nan # Not applicable, or can be set to 0 by some conventions
        p_value = 1.0 # P-value is 1.0 if there's no difference to test

    # Determine significance level for table
    sig_marker = ''
    sig_text = 'Not Significant'
    if p_value < 0.001:
        sig_marker = '***'
        sig_text = 'Significant'
    elif p_value < 0.01:
        sig_marker = '**'
        sig_text = 'Significant'
    elif p_value < 0.05:
        sig_marker = '*'
        sig_text = 'Significant'

    # Determine Interpretation/Conclusion for the table
    interpretation = "Not significantly diff. from Neutral."
    if sig_text == 'Significant':
        if median_score > neutral_point:
            interpretation = "Significantly higher than Neutral."
        elif median_score < neutral_point:
            interpretation = "Significantly lower than Neutral."
        else:
            interpretation = "Significantly different from Neutral." # General case

    # Special handling for Q11's interpretation due to negative wording
    if q_name == 'Q11':
        if sig_text == 'Significant':
            if median_score > neutral_point:
                interpretation = "Significantly high on perceived mismatch."
            elif median_score < neutral_point:
                interpretation = "Significantly low on perceived mismatch."
            else:
                interpretation = "Significantly different on perceived mismatch."
        else:
            interpretation = "Neutral on perceived mismatch."


    results_data.append({
        'Question': q_name,
        'N': n,
        'Median': median_score,
        'IQR': iqr,
        'Test Stat (W)': stat,
        'p-value': p_value,
        'Sig.': sig_marker, # Asterisks for the table
        'Interpretation/Conclusion': interpretation
    })

# Create the results DataFrame
df_results = pd.DataFrame(results_data)

# --- 3. Print the Results Table ---
print("--- Wilcoxon Signed-Rank Test Results Table ---")
# Use .to_string() for better console display without truncation
# You can copy this output directly into your LaTeX table.
print(df_results.to_string(index=False))

print("\nAnalysis complete. The table data is printed above.")