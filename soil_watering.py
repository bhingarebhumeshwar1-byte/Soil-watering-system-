import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def create_dataset(samples=300):
    """Create sample soil and weather data for training."""
    np.random.seed(42)

    data = pd.DataFrame({
        "Temperature": np.random.randint(20, 41, samples),
        "Humidity": np.random.randint(30, 91, samples),
        "Soil_Moisture": np.random.randint(10, 91, samples)
    })

    condition = (
        (data["Soil_Moisture"] < 40)
        | ((data["Temperature"] > 35) & (data["Soil_Moisture"] < 50))
    )

    data["Water_Needed"] = np.where(condition, 1, 0)

    # Small amount of noise to make the sample data less perfect.
    noise = np.random.choice([0, 1], samples, p=[0.95, 0.05])
    data["Water_Needed"] = np.bitwise_xor(
        data["Water_Needed"].astype(int), noise
    )

    return data


def train_model(data):
    """Train and return the Random Forest model."""
    features = ["Temperature", "Humidity", "Soil_Moisture"]

    X = data[features]
    y = data["Water_Needed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, y_test, predictions


def show_prediction(model, temperature, humidity, soil_moisture):
    """Predict watering requirement for given conditions."""
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Soil_Moisture": [soil_moisture]
    })

    prediction = int(model.predict(input_data)[0])

    print("\n----- Watering Decision -----")
    print(f"Temperature : {temperature:.1f} °C")
    print(f"Humidity    : {humidity:.1f} %")
    print(f"Soil Moisture: {soil_moisture:.1f} %")

    if soil_moisture < 30:
        status = "Very Dry"
    elif soil_moisture < 40:
        status = "Dry"
    elif soil_moisture < 70:
        status = "Normal"
    else:
        status = "Wet"

    print(f"Soil Status : {status}")

    if prediction == 1:
        print("Watering    : REQUIRED")
        print("Water Pump  : ON")
    else:
        print("Watering    : NOT REQUIRED")
        print("Water Pump  : OFF")

    return prediction


def show_graph(data):
    """Display soil moisture and humidity graph."""
    plt.figure(figsize=(8, 5))

    scatter = plt.scatter(
        data["Soil_Moisture"],
        data["Humidity"],
        c=data["Water_Needed"],
        cmap="bwr",
        alpha=0.7
    )

    plt.xlabel("Soil Moisture (%)")
    plt.ylabel("Humidity (%)")
    plt.title("Soil Moisture vs Humidity")
    plt.grid(True, alpha=0.3)

    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["No Water Needed", "Water Needed"])

    plt.tight_layout()
    plt.show()


def get_number(prompt, minimum, maximum):
    """Read a valid number from the user."""
    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")


def main():
    print("=" * 50)
    print("       SOIL WATERING SYSTEM")
    print("     Machine Learning Project")
    print("=" * 50)

    print("\nCreating sample dataset...")
    data = create_dataset()

    print("Training Random Forest model...")
    model, accuracy, y_test, predictions = train_model(data)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Water Needed", "Water Needed"],
            zero_division=0
        )
    )

    show_prediction(
        model,
        temperature=36,
        humidity=45,
        soil_moisture=35
    )

    while True:
        print("\nChoose an option:")
        print("1. Check watering requirement")
        print("2. Show graph")
        print("3. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            temperature = get_number(
                "Enter temperature (20-40 °C): ", 20, 40
            )
            humidity = get_number(
                "Enter humidity (30-90 %): ", 30, 90
            )
            soil_moisture = get_number(
                "Enter soil moisture (10-90 %): ", 10, 90
            )

            show_prediction(
                model,
                temperature,
                humidity,
                soil_moisture
            )

        elif choice == "2":
            show_graph(data)

        elif choice == "3":
            print("\nThank you for using the Soil Watering System.")
            break

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
