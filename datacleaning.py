"""
This section of code processes and cleans hockey game event-sequence data to prepare it 
for modeling zone exits. It normalizes spatial coordinates (x, y) to a 0-1 scale, maps 
player positions to numerical values, and scales score differentials. The code then cleans 
the zone_exit_data by ensuring all events are dictionaries, replacing NaN values with 
appropriate defaults, and applying fallback values for missing player positions, score 
differentials, and exit ratings. Finally, it creates a DataFrame from zone_exit_data, 
serializes event lists to JSON, saves successful and all sequences to CSV files, and 
generates metadata about sequence lengths and result distributions, saving everything 
to a JSON file.
"""
# Required Imports 
import pandas as pd
import json
import math 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Reading in Data 
#data = pd.read_csv("/Users/arunramji/Desktop/Linhac/Linhac24-25_Sportlogiq.csv")


data = pd.read_csv("Linhac24-25_Sportlogiq.csv")



def replace_nan(obj):
    """
    Recursively traverse the object and:
      - Replace any float 'nan' with None.
      - Convert any numpy numeric type to a native Python type.
    """
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    # Check for NumPy numeric types, e.g., np.float64, np.int64, etc.
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# Filtering for non useful events 
df2 = data[~data["eventname"].isin(["soshot","sogoal"])]

# Getting rid of b2b faceoff events for capture logic this way only the team who wins ID is visible
df2 = df2[~((df2['eventname'] == 'faceoff') & (df2['outcome'] == 'failed'))]

# Fill NaNs
columns_to_fill = ['xg_allattempts', 'teaminpossession']
df2[columns_to_fill] = df2[columns_to_fill].fillna(0)

# Filtering out end of game events as they are different then regular game flow
df2 = df2[df2["compiledgametime"] < 3570]

# Resetting index to ensure it is sequentila for my locs below 
df2 = df2.reset_index(drop=True)

# Retagging data
for index, row in df2.iterrows():
    if row["eventname"] == "controlledexit":
        # Fix for fails that are actually just uncontrolleds
        if index + 1 < len(df2) and row["outcome"] == "failed" and -25 < df2.loc[index + 1, "xadjcoord"] < 25:
            df2.loc[index, "outcome"] = "successful"
            df2.loc[index, "eventname"] = "dumpout"

        # Get rid of receptions counting as the exit and credit the passer
        elif index - 1 >= 0 and df2.loc[index - 1, "eventname"] == "reception":
            if df2.loc[index - 1, "xadjcoord"] > -25:
                # Look backwards to find the originating pass
                found = False
                i = 2  # start at -2 to skip the reception
                coord = None
                while not found and i <= 4 and index - i >= 0:
                    if df2.loc[index - i, "eventname"] == "pass":
                        found = True
                        coord = df2.loc[index - i, "xadjcoord"]
                    else:
                        i += 1
                if found:
                    df2.loc[index, "xadjcoord"] = coord
                    df2.drop(index - 1, inplace=True)  # drop NZ/OZ reception
                    df2.reset_index(drop=True, inplace=True)  # reindex

# Initializations for major loop
zone_exit_data = []
potential = []
ozoneteam = None 
inzone = False
capture = False
id = 1
possible_random = []
prevpos = None
j = 0

# Helper Functions 
# 0 means rebound or shot 
# Assigns who the the team in the ozone is and who the team in the Dzone is 
def coding_teams(oteam, seq, column):
    for line in seq:
        if line[column] == oteam or line[column] == 0:
            line[column] = 1    
        else:
            line[column] = -1   # Team under pressure

def reset_states():
    return False, False, [], [], None

def record_regular_sequence(potential, result, ozoneteam, id, zone_exit_data):
    potential = [e for e in potential if e["eventname"] not in ["controlledexit", "dumpout"]]
    coding_teams(ozoneteam, potential, "teamid")
    coding_teams(ozoneteam, potential, "teaminpossession")
    potential = [e for e in potential if not (e["teamid"] == -1 and e.get("xadjcoord", 0) > -10)]
    zone_exit_data.append({"id": id, "events": potential, "result": result})
    return id + 1

def record_random_sequence(df2, i, j, possible_random, result, ozoneteam, id, zone_exit_data):
    # Insert previous two events (before the start of random sequence)
    possible_random.insert(0, df2.iloc[i - j - 1])
    possible_random.insert(0, df2.iloc[i - j - 2])

    # Clean sequence
    possible_random = [e for e in possible_random if e["eventname"] not in ["controlledexit", "dumpout"]]
    possible_random = [e for e in possible_random if e.get("xadjcoord", 0) < -26]

    # Recode team roles
    coding_teams(ozoneteam, possible_random, "teamid")
    coding_teams(ozoneteam, possible_random, "teaminpossession")

    # Add to dataset
    zone_exit_data.append({"id": id, "events": possible_random, "result": result})
    return id + 1


# Looping through and capturing relevant data
for index, row in df2.iterrows():
    i = index
    game = row["gameid"]
    period = row["period"]

    # Different game, reset
    if i > 0 and row["gameid"] != df2.iloc[i - 1]["gameid"]:
      inzone, capture, possible_random, potential, ozoneteam = reset_states()
      continue
    # Different Period, reset
    if i > 0 and row["period"] != df2.iloc[i - 1]["period"]:
      inzone, capture, possible_random, potential, ozoneteam = reset_states()
      continue

    # No longer Even strength, reset
    if i > 0 and row["teamskatersonicecount"] != 5 or row['opposingteamskatersonicecount'] != 5:
        inzone, capture, possible_random, potential, ozoneteam = reset_states()
        continue
    
    # Tracking random exits appending values if events are from the same team
    if row["teaminpossession"] != 0:
        currpos = row["teaminpossession"]

    if row["eventname"] not in ["controlledexit", "dumpout"]:
        if row["teaminpossession"] == 0 or currpos == prevpos or prevpos == None:
            possible_random.append(row)
            j += 1
        else:
            possible_random = []
            j = 0

    if row["teaminpossession"] != 0:
        prevpos = currpos
    
    # We are now in the defensive zone
    if row["eventname"] in ["controlledentry", "dumpin"] and row["outcome"] == "successful":
        inzone = True
        ozoneteam = row["teamid"]
        potential = []

    # Defensive team has the puck, start capturing and append previous 2 events
    if row["teaminpossession"] != 0 and row["teaminpossession"] != ozoneteam and inzone == True and capture == False:
        capture = True
        potential.append(df2.iloc[i-2])
        potential.append(df2.iloc[i-1])

    # Faceoff in the dzone reset 
    if row["eventname"] == "faceoff" and inzone == True:
        potential = []
        possible_random = []
        # ozone team wins don't capture
        if row["teamid"] == ozoneteam:
            capture = False
        # Dzone team won, start capturing
        if row["teamid"] != ozoneteam:
            capture = True 
        # Weird event where immediate exit, ignore and set out of zone
        if row["type"] == "recoveredwithexit":
            capture = False
            inzone = False
    # We are now inzone
    if row["eventname"] == "icing":
        inzone = True 

    # We are inzone and dzone team has the puck, append events
    if capture == True and inzone == True and row["eventname"] not in ["controlledexit", "dumpout"]:
        potential.append(row)

    # Appending controlled exits 
    if row["eventname"] == "controlledexit":
        if row["outcome"] == "successful":
            inzone = False
            capture = False
            if len(potential) > 1:
                id = record_regular_sequence(potential, "SC", ozoneteam, id, zone_exit_data)
            elif len(potential) == 0:
                id = record_random_sequence(df2, i, j, possible_random, "SC", ozoneteam, id, zone_exit_data)

        if row["outcome"] == "failed":
            capture = False
            inzone = True
            if len(potential) > 1:
                id = record_regular_sequence(potential, "F", ozoneteam, id, zone_exit_data)
                
            elif len(potential) == 0:
                    id = record_random_sequence(df2, i, j, possible_random, "F", ozoneteam, id, zone_exit_data)
        potential = []
        possible_random = []
        j = 0

    if row["eventname"] == "dumpout":
        if row["outcome"] == "successful":
            inzone = False
            capture = False
            if len(potential) > 1:
                id = record_regular_sequence(potential, "SD", ozoneteam, id, zone_exit_data)
            elif len(potential) == 0:
                id = record_random_sequence(df2, i, j, possible_random, "SD", ozoneteam, id, zone_exit_data)

        if row["outcome"] == "failed":
            capture = False
            inzone = True
            if len(potential) > 1:
                id = record_regular_sequence(potential, "F", ozoneteam, id, zone_exit_data)
            elif len(potential) == 0:
                id = record_random_sequence(df2, i, j, possible_random, "F", ozoneteam, id, zone_exit_data)
        potential = []
        possible_random = []
        j = 0

    if capture == True and row["teaminpossession"] == ozoneteam and inzone == True:
        if len(potential) > 1:
            id = record_regular_sequence(potential, "F", ozoneteam, id, zone_exit_data)
        elif len(potential) == 0:
            id = record_random_sequence(df2, i, j, possible_random, "F", ozoneteam, id, zone_exit_data)
        capture = False
        inzone = True
        potential = []

    # Kill events 
    if row["eventname"] in ["penaltydrawn", "penalty", "goal", "offside"]:
        inzone, capture, possible_random, potential, ozoneteam = reset_states()



# Model Preprocessing 


# Normalize x and y once
df2['xadjcoord'] = (df2['xadjcoord'] + 100) / 200
df2['yadjcoord'] = (df2['yadjcoord'] + 42.5) / 85

# Map player position once
pos_map = {"D": 1, "F": 0, "G": 2}
df2["playerprimaryposition"] = df2["playerprimaryposition"].map(pos_map)

outcome_map = {"successful":1,"failed":0}
df2["outcome"] = df2["outcome"].map(outcome_map)
# Clip and scale scoredifferential once
df2["scorediff_scaled"] = df2["scoredifferential"].clip(-3, 3) / 3.0




# Make sure all events are dictionaries
for row in zone_exit_data:
    row["events"] = [dict(e) if not isinstance(e, dict) else e for e in row["events"]]

# Replace NaNs in nested dicts
for row in zone_exit_data:
    row["events"] = [replace_nan(e) for e in row["events"]]

# --- Patching  ---
outcome_map = {"successful": 1.0, "failed": 0.0}

for row in zone_exit_data:
    for ev in row["events"]:

        # Ensure playerprimaryposition is mapped (if missing)
        ev["playerprimaryposition"] = pos_map.get(
            ev.get("playerprimaryposition"), 0
        )

        # Scorediff fallback and scale
        sd = ev.get("scoredifferential", 0)
        ev["scorediff_scaled"] = max(-3, min(3, sd)) / 3.0

        # Outcome 
        outcome = ev.get("outcome")

        # Convert if it's a string
        if isinstance(outcome, str):
            outcome = outcome_map.get(outcome.lower(), 0.0)

        # Replace NaN or None
        if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
            ev["outcome"] = 0.0
        else:
            ev["outcome"] = outcome


# Now create a DataFrame from zone_exit_data
zone_df = pd.DataFrame(zone_exit_data)


# Serialize the list of dicts to a JSON string
zone_df["events"] = zone_df["events"].apply(json.dumps)

# Write to CSV

zone_df.to_csv("training2.csv", index=False)



# Build metadata
seq_lengths = zone_df["events"].apply(lambda s: len(json.loads(s)))
metadata = {
    "total_sequences": len(zone_df),
    "success_rate": f"{(zone_df['result'] == 'S').mean() * 100:.1f}%",
    "avg_seq_len": seq_lengths.mean().round(1),
    "seq_len_dist": seq_lengths.value_counts().sort_index().to_dict(),
    "result_counts": zone_df["result"].value_counts().to_dict(),
    "avg_seq_len_by_result": seq_lengths.groupby(zone_df["result"]).mean().round(1).to_dict()
}

metadata["result_percentages"] = { 
    key: f"{(val / metadata['total_sequences'] * 100):.1f}%" 
    for key, val in metadata["result_counts"].items()
}

# Save all sequences with events and labels
with open("hockey_sequences.json", "w") as f:
    json.dump({
        "sequences": zone_exit_data,
        "metadata": metadata
    }, f, indent=2)

# Uncomment if you want to see
#print(json.dumps(metadata, indent=2))
