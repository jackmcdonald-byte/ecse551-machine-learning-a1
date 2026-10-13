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

def get_rt_subcomponent(date, company_info):
    permno = company_info["permno"]
    variance_sq_index = all_date_roadmaps[permno][date]
    variance_sq=all_variance_sq_tracks[permno][variance_sq_index]
    if variance_sq == 0:
        variance_sq = small_q0
    predicted_return=company_info["predicted_return"]
    actual_return=company_info["actual_return"]
    return (actual_return * predicted_return) / (variance_sq)
def get_rt(date):
    return sum(get_rt_subcomponent(date, company_info) for company_info in calendar_portfolio[date])

def get_sharpe_ratio(date_list):

    # 1. Build the array of daily R_t returns using your function
    R_p_list = [get_rt(date) for date in date_list]
    R_p_array = np.array(R_p_list)

    # 2. Calculate your Annualized Sharpe Ratio directly from the array
    mean_R = np.mean(R_p_array)
    std_R = np.std(R_p_array)

    # Guard against division by zero if std is somehow 0
    if std_R > 0:
        sharpe_ratio = (mean_R / std_R) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return sharpe_ratio
# 1. Sort the dates chronologically so your PnL tracks properly over time
sorted_test_dates = sorted(calendar_portfolio.keys())
# 1. Group your sorted test dates into individual years
dates_2011 = [date for date in sorted_test_dates if date.startswith("2011")]
dates_2012 = [date for date in sorted_test_dates if date.startswith("2012")]
dates_2013 = [date for date in sorted_test_dates if date.startswith("2013")]

# 2. Calculate the Sharpe Ratio for each sub-period and the full timeline
sr_2011 = get_sharpe_ratio(dates_2011)
sr_2012 = get_sharpe_ratio(dates_2012)
sr_2013 = get_sharpe_ratio(dates_2013)
sr_overall = get_sharpe_ratio(sorted_test_dates) # Evaluates the full period seamlessly

#getting the necessary data for the plot
daily_returns = np.array([get_rt(d) for d in sorted_test_dates])
cumulative_pnl = np.cumsum(daily_returns)
# Print the comprehensive stability report
print(f"======================================")
print(f"    PORTFOLIO PERFORMANCE REPORT      ")
print(f"======================================")
print(f"2011 Sharpe Ratio   : {sr_2011:.4f}")
print(f"2012 Sharpe Ratio   : {sr_2012:.4f}")
print(f"2013 Sharpe Ratio   : {sr_2013:.4f}")
print(f"--------------------------------------")
print(f"Overall Sharpe Ratio: {sr_overall:.4f}")
print(f"======================================")

#plot creation
# 3. Create the plot instantly
plt.figure(figsize=(10, 5))
plt.plot(sorted_test_dates, cumulative_pnl, color="teal", label="Portfolio P&L")

# Clean up axes so dates look professional
plt.title("Cumulative Portfolio Returns (2011-2013)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) # Prevents text overlap
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

