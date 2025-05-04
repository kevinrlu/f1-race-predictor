<h1 align="center">
    F1 Race Predictions - Machine Learning Model
</h1>
<p align="center">
   Find out predicted driver placements and race times for upcoming races using machine learning.
</p>

## Important
Predictions will be available only after the qualifying session ends AND FastF1 publishes the data

## Features
- Number of circuit laps
- Qualifying times

## Quick Start
### Prerequisites
- fastf1==3.5.3
- Flask==3.1.0
- pandas==2.2.3
- scikit-learn==1.6.1

```pip install --upgrade -r requirements.txt
```

### Setup
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

### Website
- Improve UI and UX

### Circuit Features
- Circuit type (road and street)
- Direction (clockwise and counterclockwise)
- Number of DRS zones
- Number of turns

### Race Context
- Humidity (%)
- If wet, rainfall amount during the race (mm)
- Number of pit stops and average pit-stop duration (s)
- Temperatures (air and track °C)
- Tyres (tyre compound, tyre degradation, and tyre temperature)
- Weather (dry and wet)
- Wind speed and direction (km/h)

### Model Enhancements 
  - Implement more relevent models (random forest, gradient boosting)
  - Utilize more powerful learners (XGBoost, LightGBM)

## Notice
This project is unofficial and is not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trade marks of Formula One Licensing B.V.