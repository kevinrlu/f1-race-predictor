<h1 align="center">
    F1 Race Predictions - Machine Learning Model
</h1>
<p align="center">
   Find out predicted driver placements and race times for upcoming races using machine learning.
</p>

## Features
- Qualifying times
- Number of circuit laps

## Quick Start
### Prerequisites
- fastf1==3.5.3
- Flask==3.1.0
- pandas==2.2.3
- scikit-learn==1.6.1

```python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

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

### Site
- Improve ui, ux
### Circuit Features
  - Circuit type (street vs. permanent)
  - Direction (clockwise vs counterclockwise)
  - Number of turns
 - Number of DRS zones

### Race Context
  - Weather (dry / wet)
  - If wet, rainfall amount during race (mm)
  - Temperatures (air and track °C)
  - Humidity (%)
  - Wind speed and direction (km/h)
  - Number of pit stops and average pit-stop duration (s)
- tyre stuff

### Model Enhancements 
  - Implement more relevent models (random forest, gradient boosting)
  - Utilize more powerful learners (XGBoost, LightGBM)

## Notice
This project is unofficial and is not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One Licensing B.V.