## Current Features
- Qualifying times
- Number of circuit laps
- Number of DRS zones

## Future Improvements

- **Circuit Features**  
  - Circuit type (street vs. permanent)
  - Direction (clockwise vs counterclockwise)
  - Number of turns

- **Race Context**  
  - Weather (dry / wet)
  - Temperatures
  - Humid
  - Tyre compounds & stint lengths
  - Tyre degradation

- **Model Enhancements**  
  - Try interaction or polynomial features  
  - Experiment with more powerful learners (e.g. XGBoost, LightGBM)


  1. Driver & Team Performance
Qualifying performance
– Best lap time in Q1/Q2/Q3 (seconds)

Historical results
– Past finishing positions for this driver (and team) at this circuit
– Season‑to‑date average finish

Championship standings
– Driver’s current points total
– Constructor’s current points total

Driver head‑to‑head
– Recent wins/podiums vs. teammates and direct rivals

2. Circuit Characteristics
Track layout
– Total lap distance (km)
– Number of corners (and their types: high‑speed vs. low‑speed)

Overtaking difficulty
– Count/length of DRS zones
– Average historical overtakes per race

Lap count & race length
– Scheduled number of laps

3. Weather & Environmental
Ambient conditions
– Air and track temperatures (°C)
– Humidity (%)
– Wind speed and direction (km/h)

Precipitation
– Chance of rain (forecast probability)
– Actual rainfall during the race (mm)

4. Tyre & Strategy
Compound selection
– Which P Zero compounds used each stint

Pit‑stop data
– Number of stops
– Average pit‑stop duration (s)

Tyre degradation
– Estimated wear rate per lap (from practice data)

5. Race Dynamics & Incidents
Safety car likelihood
– Historical SC/VSC deployments at this track

Yellow flags
– Average number of full‑course yellows per race

Incidents & DNFs
– Driver’s season DNF rate
– Team reliability (failures per race)

6. Practice & Session Data
FP1–FP3 metrics
– Best lap times and consistency (std dev of lap times)
– Long‑run pace (average lap time over 10‑lap stints)

7. Technical & Contextual
Power unit upgrades
– Whether a new engine spec or MGU‑H was fitted (grid‑penalty flag)

Car development
– Recent aero/upgrades introduced (binary or count)

Track evolution
– Rubbering‑in rate (lap‑time improvement FP → Q)