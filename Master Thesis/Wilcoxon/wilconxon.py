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

# --- 3. Calculate Composite Median Scores for Each Category per Participant ---
df_category_scores = pd.DataFrame() # New DataFrame to store category scores

for category_name, q_list in categories.items():
    # Select columns for the current category
    category_df = df_raw[q_list]
    # Calculate the median score for each participant across the questions in this category
    # .apply(np.nanmedian, axis=1) handles potential NaNs if any, and calculates row-wise median
    df_category_scores[category_name] = category_df.apply(np.nanmedian, axis=1)

# --- 4. Perform Wilcoxon Test on Category Scores ---
results_data_category = []

for category_name in df_category_scores.columns:
    category_scores = df_category_scores[category_name]
    n = len(category_scores) # Number of participants for this category score

    median_score = category_scores.median() # Median of the *composite category scores*
    q1 = category_scores.quantile(0.25)
    q3 = category_scores.quantile(0.75)
    iqr = f"{q1:.2f} - {q3:.2f}"

    # Calculate differences from the neutral point for Wilcoxon test
    diffs = category_scores - neutral_point

    if len(diffs[diffs != 0]) > 0:
        stat, p_value = wilcoxon(diffs, alternative='two-sided', method='exact')
    else:
        stat = np.nan
        p_value = 1.0

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
            interpretation = "Significantly different from Neutral."

    # Special handling for Immersion category if Q11 is influential
    if category_name == 'Immersion':
        if sig_text == 'Significant':
            if median_score > neutral_point:
                interpretation = "Significantly higher than Neutral (indicating strong immersion)."
            elif median_score < neutral_point:
                interpretation = "Significantly lower than Neutral (indicating low immersion)."


    results_data_category.append({
        'Category': category_name,
        'N': n,
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
print("--- Wilcoxon Signed-Rank Test Results Table for Categories ---")
print(df_results_category.to_string(index=False))