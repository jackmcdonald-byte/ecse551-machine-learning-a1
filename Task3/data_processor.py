import os
import pickle
from math import sqrt
import numpy as np
from collections import Counter
from sklearn.linear_model import Ridge
# 1. Get the directory where this current script is running
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Point directly to your saved data file in the same folder
pickle_path = os.path.join(script_dir, "master_cleaned_data.pkl")

# 3. Open the file in 'rb' mode (Read Binary) and load it
with open(pickle_path, "rb") as file:
    loaded_data = pickle.load(file)
print("Success! Your master cleaned dataset is fully loaded.")
print(f"Total unique companies available: {len(loaded_data)}")
#according to google:
#must be verified
beta1 = 0.94
beta2 = 0.97
small_q0= 0.000001
all_mu_tracks = {}
all_variance_tracks = {}
all_vol_adj_return_tracks = {}
all_date_roadmaps = {}
for permno, date_dict in loaded_data.items():
    sorted_dates = sorted(date_dict.keys())
    mu_list=[]
    q_sq_list=[]
    volatility_adjusted_returns_list=[]
    date_to_index_dict ={}
    for i in range(len(sorted_dates)):
        rt=date_dict[sorted_dates[i]]["returns"]
        date_to_index_dict[sorted_dates[i]] = i
        #computing mu
        if i == 0:
            mu_list.append(date_dict[sorted_dates[i]]["returns"])
        else:
            mu=beta2*mu_list[i-1]+(1-beta2)*rt
            mu_list.append(mu)
        #computing yesterday's volatility
        if i == 0:
            yesterday_vol = sqrt(small_q0)
        else:
            # Look back at the variance that was appended on the previous loop step
            yesterday_vol = sqrt(q_sq_list[i-1])
            
        # Handle extreme edge case to avoid division by zero
        if yesterday_vol == 0:
            yesterday_vol = sqrt(small_q0)
        #computing volatility-adjusted returns
        volatility_adjusted_returns_list.append(rt/yesterday_vol)
        #computing q
        mt=mu_list[i]
        at=rt-mt
        if i == 0:
            q_sq_list.append(small_q0)
        else:
            q=beta1*q_sq_list[i-1]+(1-beta1)*(at**2)
            q_sq_list.append(q)

    all_mu_tracks[permno] = mu_list
    all_variance_tracks[permno] = q_sq_list
    all_vol_adj_return_tracks[permno] = volatility_adjusted_returns_list
    all_date_roadmaps[permno] = date_to_index_dict
# 1. Initialize your strict Train and Test compartments
date_stock_counts = Counter(
    date for date_dict in loaded_data.values() for date in date_dict.keys()
)
X_train, Y_train, meta_train = [], [], []
X_test, Y_test, meta_test = [], [], []
Y_test_raw = []  # This will hold the raw unadjusted returns for your portfolio P&L scoring
start_year_test = 2011
minimum_stock_threshold = 100  # Minimum number of stocks required for a date to be included in the test set
for permno, date_dict in loaded_data.items():
    sorted_dates=sorted(date_dict.keys())
    #the loop will ignore the last day because
    #we cannot predict the next day return if we are on the last day
    #also starts from 1 because we need the info
    #until yesterday to predict tomorrow's return
    for i in range(1, len(sorted_dates)-1):
        current_date=sorted_dates[i]
        # 🚨 YOUR FILTER LAYER: Enforce the explicit constraint instantly 🚨
        # If this date is in the test window AND doesn't have 100 stocks, skip it completely!
        year = int(current_date.split('-')[0])
        if year >= start_year_test and date_stock_counts[current_date] < minimum_stock_threshold:
            continue  # Move to the next loop step; does not append anything to test lists
        yesterday_date=sorted_dates[i-1]
        tomorrow_date=sorted_dates[i+1]
        current_day_index=all_date_roadmaps[permno][current_date]
        yesterday_day_index=all_date_roadmaps[permno][yesterday_date]
        tomorrow_day_index=all_date_roadmaps[permno][tomorrow_date]
        #get yesterday's info
        yesterday_mu=all_mu_tracks[permno][yesterday_day_index]
        yesterday_variance=all_variance_tracks[permno][yesterday_day_index]
        yesterday_vol_adj_return=all_vol_adj_return_tracks[permno][yesterday_day_index]
        yesterday_net_sentiment=loaded_data[permno][yesterday_date]["bullish"]-loaded_data[permno][yesterday_date]["bearish"]
        feature_vector=[yesterday_mu, yesterday_variance, yesterday_vol_adj_return, yesterday_net_sentiment]
        #get tomorrow's info
        tomorrow_vol_adj_return=all_vol_adj_return_tracks[permno][tomorrow_day_index]
        if year < start_year_test:
            X_train.append(feature_vector)
            Y_train.append(tomorrow_vol_adj_return)
            meta_train.append((permno, current_date))
        else:
            X_test.append(feature_vector)
            Y_test.append(tomorrow_vol_adj_return)
            meta_test.append((permno, current_date))
            # Capture the raw unadjusted return specifically for your portfolio P&L scoring
            Y_test_raw.append(loaded_data[permno][tomorrow_date]["returns"])
#make numpy arrays for the features and labels
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test,  Y_test  = np.array(X_test),  np.array(Y_test)
#model training using ridge regression
model = Ridge(alpha=1.0)
print("Training Ridge Regression model on 2009-2010 data...")
model.fit(X_train, Y_train)
print("Training complete!")
# Order: [yesterday_mu, yesterday_variance, yesterday_vol_adj_return, yesterday_net_sentiment]
print(f"Learned Feature Coefficients: {model.coef_}")

print("Generating return predictions for the 2011-2013 test window...")
test_predictions = model.predict(X_test)

calendar_portfolio={}
for idx, (permno, date) in enumerate(meta_test):
    if date not in calendar_portfolio:
        calendar_portfolio[date] = []
    calendar_portfolio[date].append({
        "permno":permno,
        "predicted_return":test_predictions[idx],
        "actual_return":Y_test_raw[idx]
    })

#packaging the data for the next task
workspace_path = os.path.join(script_dir, "trained_model.pkl")
data={
    "calendar_portfolio":calendar_portfolio,
    "all_date_roadmaps":all_date_roadmaps,
    "all_variance_tracks":all_variance_tracks,
    "all_mu_tracks":all_mu_tracks,
    "all_vol_adj_return_tracks":all_vol_adj_return_tracks,
    "small_q0":small_q0,
    "beta1":beta1,
    "beta2":beta2,
    "model":model
}
with open(workspace_path, "wb") as f:
    pickle.dump(data, f)

print(f"\n🎉 SUCCESS! Full workspace safely serialized to: {workspace_path}")