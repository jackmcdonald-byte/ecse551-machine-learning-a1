import os
import pandas as pd
from datetime import datetime, timedelta
import pickle
# 1. Get the exact folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def csv_to_date_dict(filename):
    """Loads a wide stock CSV file into a nested dictionary using absolute paths."""
    # 2. Combine the script directory with your filename
    full_path = os.path.join(script_dir, filename)
    
    df = pd.read_csv(full_path)
    nested_dict = df.set_index("date").to_dict(orient="index")
    return nested_dict


def get_next_monday(date_str):
    """
    Takes a 'YYYY-MM-DD' date string.
    If it's a weekend, returns the string for the upcoming Monday.
    If it's a weekday, returns the original date string unchanged.
    """
    # 1. Convert the text string into a real Python date object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    # 2. Get the name of the day (e.g., 'Saturday', 'Monday')
    day_name = date_obj.strftime('%A')
    
    # 3. Check if it's a weekend and add the correct number of days
    if day_name == "Saturday":
        next_monday = date_obj + timedelta(days=2)
        return next_monday.strftime("%Y-%m-%d")
    elif day_name == "Sunday":
        next_monday = date_obj + timedelta(days=1)
        return next_monday.strftime("%Y-%m-%d")
    
    # 4. If it's a weekday, return it as 
    return None


# 3. Create your separate dictionaries safely
indicator_dict = csv_to_date_dict("eq_indicator.csv")
returns_dict = csv_to_date_dict("eq_returns.csv")
bullish_dict = csv_to_date_dict("psych_bullish.csv")
bearish_dict = csv_to_date_dict("psych_bearish.csv") 
# Quick dictionary comprehension for all 4 dictionaries
indicator_dict = {d: v for d, v in indicator_dict.items() if 2009 <= int(d.split('-')[0]) <= 2013}
returns_dict   = {d: v for d, v in returns_dict.items() if 2009 <= int(d.split('-')[0]) <= 2013}
new_bullish_dict   = {d: v for d, v in bullish_dict.items() if 2009 <= int(d.split('-')[0]) <= 2013}
new_bearish_dict   = {d: v for d, v in bearish_dict.items() if 2009 <= int(d.split('-')[0]) <= 2013}

# Loop through the dates (since bullish and bearish share the exact same date keys)
for date in list(new_bullish_dict.keys()):
    new_date = get_next_monday(date)
    
    # If it is a weekend, we have a target Monday to roll data into
    if new_date:
        # Unpack the companies for this specific weekend date
        for permno in new_bullish_dict[date].keys():

            # --- 1. PROCESS BULLISH DATA ---
            bull_score = new_bullish_dict[date][permno]
            if not pd.isna(bull_score):
                # Safely check if the target Monday has a valid number or NaN
                if not pd.isna(new_bullish_dict[new_date][permno]):
                    new_bullish_dict[new_date][permno] += bull_score
                else:
                    new_bullish_dict[new_date][permno] = bull_score
                    
            # --- 2. PROCESS BEARISH DATA ---
            bear_score = new_bearish_dict[date][permno]
            if not pd.isna(bear_score):
                # Safely check if the target Monday has a valid number or NaN
                if not pd.isna(new_bearish_dict[new_date][permno]):
                    new_bearish_dict[new_date][permno] += bear_score
                else:
                    new_bearish_dict[new_date][permno] = bear_score
        #putting None in argument of pop() will not raise an error if the key is not found
        # , it will just return None instead of raising a KeyError. 
        
        new_bullish_dict.pop(date, None)
        new_bearish_dict.pop(date, None)




# Initialize your empty master dictionary
cleaned_data = {}

for date, permnos in indicator_dict.items():
    # 0. WEEKEND FILTER LAYER: If get_next_monday returns a date, it means today IS a weekend!
    if get_next_monday(date) is not None:
        continue  # Skip this day completely
        
    for permno in permnos.keys():
        # 1. Grab the index membership indicator status
        is_in_sp500 = indicator_dict[date][permno]
        
        # 2. FILTER LAYER: Skip the company completely if it wasn't in the S&P 500 on this day
        if is_in_sp500 != 1:
            continue
            
        # 3. Grab the return value
        ret_val = returns_dict[date][permno]
        
        # 4. FILTER LAYER: Skip if the return value itself is missing (e.g., market holidays)
        if pd.isna(ret_val):
            continue
            
        # 5. SAFE LOOKUP LAYER: Use .get() to pull sentiment, safely defaulting to 0.0 if missing
        bull_score = new_bullish_dict.get(date, {}).get(permno, 0.0)
        bear_score = new_bearish_dict.get(date, {}).get(permno, 0.0)
        
        # 6. Change any remaining NaN sentiment values to 0.0 so your model doesn't crash later
        if pd.isna(bull_score): bull_score = 0.0
        if pd.isna(bear_score): bear_score = 0.0
        
        # 7. INITIALIZATION LAYER: If this company is new to our dict, give it its own sub-dictionary
        if permno not in cleaned_data:
            cleaned_data[permno] = {}
            
        # 8. Store the clean data point inside your structure using [permno][date]
        cleaned_data[permno][date] = {
            "returns": float(ret_val),
            "bullish": float(bull_score),
            "bearish": float(bear_score)
        }



#importing to pickle
# 1. Define the name you want for your saved data file
pickle_path = os.path.join(script_dir, "master_cleaned_data.pkl")

# 2. Open the file in 'wb' mode (Write Binary) and save the dictionary
with open(pickle_path, "wb") as file:
    pickle.dump(cleaned_data, file)

print(f"Data cleanup complete! Saved master file safely to: {pickle_path}")
        


