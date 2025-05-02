<h1 align="center">
    F1 Race Predictions - Machine Learning Model
</h1>
<p align="center">
   Find out predicted driver placements and race times for upcoming races using machine learning
</p>

## Features



## Quick Start
### Prerequisites
- Flask==2.2.5
- fastf1==2.3.1
- pandas==2.0.1
- scikit-learn==1.2.2

``` pip install -U -r requirements.txt```

### Setup: NOTE PREDICTIONS ARE ONLY AVAILABLE AFTER RACE QUALIFIERS
```mkdir f1_cache
python app.py
```

## Architecture

### Frontend
- **HTML**
- **CSS**

### Backend
- **Python**
- **Flask**
- **FastF1**
- **scikit-learn**

## Future Improvements

## Current Features
- Qualifying times
- Number of circuit laps
- Number of DRS zones

## Future Improvements

###Circuit Features
  - Circuit type (street vs. permanent)
  - Direction (clockwise vs counterclockwise)
  - Number of turns

###Race Context
  - Weather (dry / wet)
  - If wet, rainfall amount during race (mm)
  - Temperatures (air and track °C)
  - Humidity (%)
  - Tyre compound selection & stint lengths
  - Tyre degradation
  - Wind speed and direction (km/h)
  - Number of pit stops and average pit-stop duration (s)

###Model Enhancements 
  - Try interaction or polynomial features  
  - Experiment with more powerful learners (e.g. XGBoost, LightGBM)

  tailwind and do something about html

Incidents & DNFs
– Driver’s season DNF rate
– Team reliability (failures per race)

6. Practice & Session Data
FP1–FP3 metrics
– Best lap times and consistency (std dev of lap times)
– Long‑run pace (average lap time over 10‑lap stints)
