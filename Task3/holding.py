import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
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

#getting each companies predictiors 
sorted_test_dates = sorted(calendar_portfolio.keys())
company_histories={}
#Note: because the traversal is chronological, the predictions are 
#also chronological
for date in sorted_test_dates:
    for company_info in calendar_portfolio[date]:
        prediction=company_info["predicted_return"]
        permno=company_info["permno"]
        if permno not in company_histories:
            company_histories[permno]=[]
        company_histories[permno].append(prediction)

# getting the inner ratio for each company
company_holding_dict = {}
for permno, predictons in company_histories.items():
    predictons_array=np.array(predictons)
    T=len(predictons)
    if T>1:
        numerator=np.sum(np.abs(predictons_array))
        denominator=np.sum(np.abs(np.diff(predictons_array)))
        if denominator > 0:
            company_holding_dict[permno]=numerator/denominator
# Extract all the calculated individual company ratios
holding_period_ratios = list(company_holding_dict.values())

# Calculate the final portfolio average (Equation 7)
H_av = np.mean(holding_period_ratios)

print("======================================")
# Bold highlights the definitive answer and core figures directly
print(f"**Final Average Holding Period (H_av): {H_av:.2f} Trading Days**")
print(f"Total Unique Equities Evaluated (N): {len(holding_period_ratios)}")
print("======================================")