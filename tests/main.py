from src.app import SolfaLearningApp
from src.data_collector import SolfaDataCollector
from src.model_trainer import train_model

def collect_training_data():
    collector = SolfaDataCollector()
    gestures = ['do', 'di', 'ra', 're', 'ri', 'me', 'mi', 'fa', 'fi', 'se', 'so', 'si', 'le', 'la', 'li', 'te', 'ti']
    for gesture in gestures:
        print(f"Collecting data for {gesture}...")
        collector.collect_data(gesture, num_samples=200)
    collector.save_data()

def main():
    # Uncomment these lines when you need to collect new data or retrain the model
    collect_training_data()
    train_model()

    app = SolfaLearningApp()
    app.run()

if __name__ == "__main__":
    main()
