from confluent_kafka import Producer
import pandas as pd
import time
import json
import random

# Kafka configuration
kafka_config = {'bootstrap.servers': 'localhost:9092'}

# Kafka topic
topic = 'test-topic'

# CSV file
csv_file = 'engines_dataset-train.csv'

# Initialize Kafka producers for each sensor
producers = {
    "engineRpm": Producer(kafka_config),
    "lubOilPressure": Producer(kafka_config),
    "fuelPressure": Producer(kafka_config),
    "coolantPressure": Producer(kafka_config),
    "lubOilTemp": Producer(kafka_config),
    "coolantTemp": Producer(kafka_config),
}

def delivery_report(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def simulate_sensor_data():
    """Simulates different sensor producers sending data independently."""
    
    # Load and clean the CSV
    data = pd.read_csv(csv_file)

    # Standardize column names (strip spaces, match expected names)
    data.columns = data.columns.str.strip()

    # Rename columns if necessary
    column_mapping = {
        "Engine rpm": "engineRpm",
        "Lub oil pressure": "lubOilPressure",
        "Fuel pressure": "fuelPressure",
        "Coolant pressure": "coolantPressure",
        "lub oil temp": "lubOilTemp",
        "Coolant temp": "coolantTemp",
    }
    
    data.rename(columns=column_mapping, inplace=True)

    print("Starting sensor data simulation...")
    print("Using multi-class classification mode for parameter evaluation")

    for index, row in data.iterrows():
        timestamp = int(time.time())  # Current timestamp

        # Convert row data to dictionary
        row_dict = row.to_dict()

        # 20% chance of skipping 1-2 random sensors
        if random.random() < 0.2:  
            missing_sensors = random.sample(list(producers.keys()), k=random.randint(1, 2))
        else:
            missing_sensors = []

        for sensor, producer in producers.items():
            if sensor in missing_sensors:
                print(f"Skipping {sensor} for timestamp {timestamp}")  
                continue
            
            # Create message payload
            message = {
                "sensor_id": sensor,
                "timestamp": timestamp,
                "value": row_dict.get(sensor, None)  # Use .get() to avoid KeyError
            }

            # Convert message to JSON
            sensor_json = json.dumps(message)

            # Send data to Kafka
            producer.produce(topic, value=sensor_json, callback=delivery_report)

            producer.flush()
            print(f"Sent data: {sensor_json}")

        time.sleep(2)  # Simulate real-time data streaming

    print("Sensor data simulation complete.")

if __name__ == '__main__':
    try:
        simulate_sensor_data()
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        # Flush all producers
        for producer in producers.values():
            producer.flush()