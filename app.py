import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hotel_model.pkl"

model = None
model_error = None


# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    model_error = str(error)


FEATURE_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_nights",
    "family_size",
    "is_family",
    "has_previous_cancel",
    "room_changed",
]


def get_form_value(field_name, value_type=str):
    """Read and validate a value from the form."""

    value = request.form.get(field_name, "").strip()

    if value == "":
        raise ValueError(f"{field_name} is required.")

    try:
        return value_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid value for {field_name}."
        ) from error


def create_input_dataframe():
    """Create the model input DataFrame."""

    hotel = get_form_value("hotel")
    lead_time = get_form_value("lead_time", int)
    arrival_date = get_form_value("arrival_date")

    arrival = pd.to_datetime(arrival_date)

    arrival_year = arrival.year
    arrival_month = arrival.strftime("%B")
    arrival_week = int(arrival.isocalendar().week)
    arrival_day = arrival.day
    weekend_nights = get_form_value(
        "stays_in_weekend_nights",
        int,
    )

    week_nights = get_form_value(
        "stays_in_week_nights",
        int,
    )

    adults = get_form_value("adults", int)
    children = get_form_value("children", float)
    babies = get_form_value("babies", int)

    meal = get_form_value("meal")
    country = get_form_value("country").upper()
    market_segment = get_form_value("market_segment")

    distribution_channel = get_form_value(
        "distribution_channel"
    )

    deposit_type = get_form_value("deposit_type")
    adr = get_form_value("adr", float)

    special_requests = get_form_value(
        "total_of_special_requests",
        int,
    )

    numeric_values = [
        lead_time,
        weekend_nights,
        week_nights,
        adults,
        children,
        babies,
        adr,
        special_requests,
    ]

    if any(value < 0 for value in numeric_values):
        raise ValueError("Values cannot be negative.")

    if adults + children + babies <= 0:
        raise ValueError(
            "The booking must contain at least one guest."
        )

    # Derived features
    total_nights = weekend_nights + week_nights
    family_size = adults + children + babies
    is_family = int(family_size > 1)

    data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "arrival_date_week_number": arrival_week,
        "arrival_date_day_of_month": arrival_day,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": deposit_type,
        "agent": 0.0,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": adr,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": special_requests,
        "total_nights": total_nights,
        "family_size": family_size,
        "is_family": is_family,
        "has_previous_cancel": 0,
        "room_changed": 0,
    }

    dataframe = pd.DataFrame([data])

    return dataframe[FEATURE_COLUMNS]


def get_cancellation_probability(dataframe):
    """Return cancellation probability."""

    if not hasattr(model, "predict_proba"):
        prediction = int(model.predict(dataframe)[0])
        return float(prediction)

    probabilities = model.predict_proba(dataframe)[0]
    classes = list(getattr(model, "classes_", []))

    if 1 in classes:
        index = classes.index(1)
        return float(probabilities[index])

    if "1" in classes:
        index = classes.index("1")
        return float(probabilities[index])

    if len(probabilities) == 2:
        return float(probabilities[1])

    raise ValueError(
        "The cancellation class could not be identified."
    )


@app.route("/", methods=["GET", "POST"])
def home():
    """Display the page and process predictions."""

    # Display the page
    if request.method == "GET":
        return render_template(
            "index.html",
            model_error=model_error,
        )

    # Check whether the model loaded successfully
    if model is None:
        return render_template(
            "index.html",
            model_error=(
                "The model could not be loaded. "
                "Make sure hotel_model.pkl is beside app.py. "
                f"Details: {model_error}"
            ),
            form_data=request.form,
        ), 503

    try:
        dataframe = create_input_dataframe()

        prediction = int(model.predict(dataframe)[0])

        probability = get_cancellation_probability(
            dataframe
        )

        percentage = round(probability * 100, 1)
        high_risk = prediction == 1

        result = {
            "risk": (
                "High Risk"
                if high_risk
                else "Low Risk"
            ),
            "risk_class": (
                "high-risk"
                if high_risk
                else "low-risk"
            ),
            "icon": "!" if high_risk else "✓",
            "probability": f"{percentage:.1f}%",
            "probability_value": percentage,
            "message": (
                "This booking is likely to be cancelled."
                if high_risk
                else
                "This booking is unlikely to be cancelled."
            ),
        }

        return render_template(
            "index.html",
            result=result,
            form_data=request.form,
        )

    except Exception as error:
        return render_template(
            "index.html",
            error=f"Prediction failed: {error}",
            form_data=request.form,
        ), 400


@app.route("/health", methods=["GET"])
def health():
    """Application health check."""

    if model is None:
        return {
            "status": "error",
            "message": "Model is not loaded",
            "details": model_error,
        }, 503

    return {
        "status": "ok",
        "message": "Application and model are ready",
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
