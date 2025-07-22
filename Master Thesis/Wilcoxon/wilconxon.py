import pandas as pd
import numpy as np
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

# --- 2. Define Categories and Map Questions ---
categories = {
    'Hand Ownership and Control': ['Q1', 'Q3', 'Q4', 'Q5', 'Q8'],
    'Realistic and Aligned Tactile Sensations Felt': ['Q2', 'Q6', 'Q7', 'Q9', 'Q10'],
    'Immersion': ['Q11', 'Q12']
}

# --- 3. Perform Wilcoxon Test on Individual Responses for Each Category ---
results_data_category = []

for category_name, q_list in categories.items():
    # from all participants, and flatten them into one list.
    pooled_responses = df_raw[q_list].values.flatten()
    pooled_responses = pooled_responses[~np.isnan(pooled_responses)] # Remove any NaNs if present

    n_value_for_table = len(pooled_responses) # N is now the total count of individual responses for the category

    median_score = np.median(pooled_responses) # Median of the *pooled* individual responses
    q1 = np.quantile(pooled_responses, 0.25)
    q3 = np.quantile(pooled_responses, 0.75)
    iqr = f"{q1:.2f} - {q3:.2f}"

    # Calculate differences from the neutral point for Wilcoxon test
    diffs = pooled_responses - neutral_point

    # Perform Wilcoxon test on the pooled data
    # (Acknowledging the statistical caveat about independence for this specific test)
    if len(diffs[diffs != 0]) > 0:
        stat, p_value = wilcoxon(diffs, alternative='two-sided', method='exact')
    else:
        stat = np.nan
        p_value = 1.0 # Or appropriate value if no non-zero diffs

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

    interpretation = "Not significantly diff. from Neutral."
    if sig_text == 'Significant':
        if median_score > neutral_point:
            interpretation = "Significantly higher than Neutral."
        elif median_score < neutral_point:
            interpretation = "Significantly lower than Neutral."
        else:
            interpretation = "Significantly different from Neutral."


    results_data_category.append({
        'Category': category_name,
        'N': n_value_for_table, # This N is now the total pooled responses
        'Median': median_score,
        'IQR': iqr,
        'Test Stat (W)': stat,
        'p-value': p_value,
        'Sig.': sig_marker,
        'Interpretation/Conclusion': interpretation
    })

# Create the results DataFrame for categories
df_results_category = pd.DataFrame(results_data_category)

# Print the results DataFrame for categories
print("--- Wilcoxon Signed-Rank Test Results Table for Categories (Pooled Individual Responses) ---")
print(df_results_category.to_string(index=False))