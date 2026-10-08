# Soil Watering System

A simple machine-learning based soil watering decision system developed in Python.

The system uses:
- Temperature
- Humidity
- Soil Moisture

A Random Forest Classifier predicts whether watering is required.

## Features

- Sample soil and weather dataset generation
- Random Forest machine learning model
- Model accuracy and classification report
- Watering prediction
- Soil condition detection
- User input mode
- Data visualization
- No external database or hardware required

## Project Structure

```text
Soil_Watering_System/
├── soil_watering.py
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9 or newer is recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python soil_watering.py
```

The program trains the model and then provides a menu:

```text
1. Check watering requirement
2. Show graph
3. Exit
```

Enter temperature, humidity, and soil moisture values to get the watering decision.

## Example

For:

```text
Temperature: 36 °C
Humidity: 45 %
Soil Moisture: 35 %
```

the system can produce:

```text
Soil Status : Dry
Watering    : REQUIRED
Water Pump  : ON
```

## Important Note

This project currently simulates the pump decision in software. It does not physically switch a water pump.

For a hardware version, the model can later be connected to an ESP32 or Arduino with a soil-moisture sensor and relay module.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Random Forest
