import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
# 1. Get the directory where this current script is running
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Point directly to your saved data file in the same folder
pickle_path = os.path.join(script_dir, "trained_model.pkl")

# 3. Open the file in 'rb' mode (Read Binary) and load it
with open(pickle_path, "rb") as file:
    loaded_data = pickle.load(file)
print("Success! Your trained Ridge regression model is fully loaded.")

#unloading the data from the pickle file
calendar_portfolio=loaded_data["calendar_portfolio"]
all_date_roadmaps=loaded_data["all_date_roadmaps"]
all_variance_sq_tracks=loaded_data["all_variance_sq_tracks"]
all_mu_tracks=loaded_data["all_mu_tracks"]
all_vol_adj_return_tracks=loaded_data["all_vol_adj_return_tracks"]
small_q0=loaded_data["small_q0"]
beta1=loaded_data["beta1"]
beta2=loaded_data["beta2"]
model=loaded_data["model"]
#getting the data for this part
sorted_test_days=sorted(calendar_portfolio.keys())
daily_correlations = []
valid_dates_tracked = []

for date in sorted_test_days:
    company_informations=calendar_portfolio[date]
    predictions=[information["predicted_return"] for information in company_informations]
    actuals=[information["actual_return"] for information in company_informations]

    if len(predictions)>1:
        #spearmanr returns a tuple: (correlation_coefficient, p_value)
        coefficent,_=spearmanr(predictions,actuals)
        if not np.isnan(coefficent):
            daily_correlations.append(coefficent)
            valid_dates_tracked.append(date)
#output of results
cumulative_rank_correlation=np.cumsum(daily_correlations)

print("======================================")
print("   RANK CORRELATION DIAGNOSTICS       ")
print("======================================")
print(f"Average Daily Rank Correlation: {np.mean(daily_correlations):.6f}")
print(f"Final Cumulative Rank Score   : {cumulative_rank_correlation[-1]:.4f}")
print("======================================")

#plotting
plt.figure(figsize=(11, 5))
plt.plot(valid_dates_tracked, cumulative_rank_correlation, color="darkslateblue", linewidth=2)

plt.title("Cumulative Daily Spearman Rank Correlation (2011-2013)", fontsize=13, fontweight="bold")
plt.xlabel("Date", fontsize=11)
plt.ylabel("Cumulative Rank Correlation Coefficient", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)

# Prevent date text overlapping on bottom axis
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
