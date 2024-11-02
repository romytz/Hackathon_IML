# Hackathon IML 🚍

## Project Overview 📊

This project was developed during a hackathon focused on **Intelligent Machine Learning (IML)** applications in transportation. The goal was to leverage data analysis and machine learning to predict key metrics for a public transportation system, specifically focusing on passenger boarding and trip duration.

### Key Objectives:
1. **Passenger Boarding Prediction**: Predict the number of passengers boarding at a specific bus stop.
2. **Trip Duration Prediction**: Estimate the duration of a bus trip from the first station to the last.

### Features:
- **Data Preprocessing**: Includes steps for handling missing values, encoding categorical features, and feature engineering.
- **Model Training**: Utilizes machine learning models such as `RandomForestRegressor` and `XGBRegressor` for regression tasks.
- **Evaluation**: Assesses model performance using metrics like Mean Squared Error (MSE).

## Installation and Setup 🛠️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/romytz/Hackathon_IML.git
   cd Hackathon_IML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure 📁

- **`data/`**: Contains raw and processed datasets.
- **`report/`**: Holds any reports generated during analysis.
- **`main.py`**: Main module for running the project.
- **`preprocessing_passenger_boardings.py`**: Script for preprocessing the passenger boarding data.
- **`preprocessing_trip_duration.py`**: Script for preprocessing trip duration data.
- **`eval_passengers_up.py`**: Evaluation script for passenger boarding predictions.
- **`eval_trip_duration.py`**: Evaluation script for trip duration predictions.
- **`Add_duration_to_all_data.py`**: Adds trip duration to the dataset for model training.
- **`plotting.py`**: Helper module for generating visualizations.

## How to Use 🖥️

1. **Run the Main Script**:
   The `main.py` script handles both data preprocessing and model evaluation based on the specified task. Specify either `passenger_boardings` or `trip_duration` as the task.

   ```bash
   python main.py --task <task> --training_set <path_to_training_set> --test_set <path_to_test_set> --out <output_file>
   ```

   - `<task>`: Choose `passenger_boardings` to predict passenger numbers or `trip_duration` to predict trip duration.
   - `<path_to_training_set>`: Path to the training dataset.
   - `<path_to_test_set>`: Path to the test dataset.
   - `<output_file>`: File path to save the output.

2. **Task-Specific Details**:
   - **Passenger Boardings Prediction**: Preprocessing and model evaluation for passenger boardings is handled within the `eval_passengers_up.py` module.
   - **Trip Duration Prediction**: Preprocessing (adding trip duration information) is handled by `Add_duration_to_all_data.py`, and model evaluation is handled within the `eval_trip_duration.py` module.

## Results 📈

- **Passenger Boarding Prediction**: The model achieved an MSE of 0.09.
- **Trip Duration Prediction**: The model achieved an MSE of 13.36.

## Future Improvements 🚀

- **Feature Engineering**: Incorporate additional features such as weather data or event schedules that could impact boarding numbers and trip duration.
- **Model Optimization**: Experiment with hyperparameter tuning and alternative machine learning algorithms to improve accuracy.

## Contributors 👥

- **Romy Tzafrir** - [GitHub Profile](https://github.com/romytz)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

