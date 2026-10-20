# Seoul Bike Rental Demand Prediction — Machine Learning and Deep Learning

A regression-based Machine Learning and Deep Learning project focused on predicting bicycle rental demand using weather and environmental features from the Seoul Bike Sharing Dataset.

The project explores relationships between weather conditions and bicycle rental counts, implements Linear Regression and TensorFlow neural networks, and compares their predictive performance.

## Project Overview

Bike-sharing systems depend on understanding rental demand to support bicycle availability and resource planning.

This project investigates how environmental conditions relate to bicycle rental demand in Seoul, South Korea.

The analysis focuses specifically on **bicycle rentals recorded at 12:00 PM (noon)**.

### Objectives

- Explore relationships between weather conditions and bike rental demand.
- Visualize feature distributions and relationships.
- Implement simple and multiple Linear Regression.
- Build regression models using TensorFlow/Keras.
- Compare traditional regression models with neural networks.
- Evaluate predictions using regression metrics.

## Dataset

**Dataset:** Seoul Bike Sharing Demand Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

The dataset contains hourly bicycle rental counts and related weather information.

### Features Used

| Feature | Description |
|---|---|
| bike_count | Number of rented bicycles |
| temp | Temperature |
| humidity | Humidity |
| wind | Wind speed |
| visibility | Visibility |
| dew_pt_temp | Dew point temperature |
| radiation | Solar radiation |
| rain | Rainfall |
| snow | Snowfall |
| functional | Whether the bike-sharing system was functioning |

**Target Variable:** `bike_count`

The notebook filters the dataset to observations recorded at noon and removes the hour column.

After exploratory analysis, wind speed, visibility, and functional status are excluded from the final regression feature set.

The multiple regression experiments use six input features: temperature, humidity, dew point temperature, solar radiation, rainfall, and snowfall.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Development Environment | Google Colab, Jupyter Notebook |

## Project Workflow

### 1. Data Loading and Preprocessing

- Loaded the Seoul Bike Sharing dataset.
- Renamed columns for easier analysis.
- Converted functional status into a binary representation.
- Filtered records to 12:00 PM.
- Removed unused columns.
- Prepared training, validation, and testing subsets.

### 2. Exploratory Data Analysis

Created scatter plots to examine relationships between bicycle rental counts and environmental features.

The analysis explored:

- Temperature
- Humidity
- Wind speed
- Visibility
- Dew point temperature
- Solar radiation
- Rainfall
- Snowfall

### 3. Simple Linear Regression

Implemented Linear Regression using temperature as the only input feature.

The model estimates bike rental demand based on the relationship between temperature and rental counts.

**Test R² Score:** 0.2799

A regression line was plotted against the observed data to visualize the fitted relationship.

### 4. Multiple Linear Regression

Extended the regression model to use six environmental features.

**Test R² Score:** 0.4013

The model incorporates multiple weather variables rather than relying only on temperature.

### 5. Neural Network Regression

Implemented feedforward neural networks using TensorFlow/Keras.

Experiments included:

- A single Dense layer for temperature-based regression.
- A deeper neural network with multiple hidden layers.
- A multilayer neural network using six environmental features.

The neural network experiments used:

- ReLU activation in hidden layers.
- Linear output for regression.
- Adam optimizer.
- Mean Squared Error loss.
- Training and validation loss visualization.

## Model Performance

The notebook compares multiple Linear Regression with a neural network trained on the same six input features.

| Model | Test MSE |
|---|---:|
| Multiple Linear Regression | 119,323.81 |
| Neural Network | 92,952.97 |

The neural network achieved a lower Mean Squared Error in this experiment.

### Performance Interpretation

- Linear Regression provides a simple and interpretable baseline.
- Neural networks can model nonlinear relationships between weather conditions and bicycle rental demand.
- The neural network achieved lower prediction error on the recorded test split.
- Additional evaluation would be required to determine whether this improvement generalizes consistently.

## Visualizations

The notebook includes:

- Scatter plots of weather features against bicycle rental counts.
- Linear Regression fitted-line visualization.
- Neural network prediction curves.
- Training and validation loss plots.
- Actual versus predicted rental count comparisons.

## Project Structure

```text id="k23t0p"
bikes-regression/
│
├── bikes_regression.ipynb
├── SeoulBikeData.csv
├── README.md
└── LICENSE
```

## Installation and Usage

### 1. Clone the Repository

```bash id="bik1cl"
git clone https://github.com/Radhika-238/bikes-regression.git
cd bikes-regression
```

### 2. Install Dependencies

```bash id="bik2dp"
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### 3. Run the Notebook

Open `bikes_regression.ipynb` using Jupyter Notebook or Google Colab.

Ensure `SeoulBikeData.csv` is available in the notebook's working directory.

Execute the notebook cells sequentially to reproduce the analysis, training, and evaluation.

## Key Learnings

This project provided practical experience in:

- Preparing real-world regression datasets.
- Exploring relationships between environmental features and numerical targets.
- Implementing simple and multiple Linear Regression.
- Understanding regression coefficients and R².
- Building neural networks using TensorFlow/Keras.
- Experimenting with different neural network architectures.
- Monitoring training and validation loss.
- Comparing models using Mean Squared Error.
- Visualizing predicted versus actual values.

## Limitations

- The analysis focuses only on noon observations rather than all hourly rental demand.
- Feature selection was based on exploratory visual analysis rather than systematic validation.
- The notebook uses random train-validation-test splitting, which may not fully account for chronological patterns.
- Results are based on a particular experimental split.
- Some normalization layers are initialized but not included in the final neural network architectures.
- Further experiments are needed to assess generalization and reproducibility.

## Future Improvements

- Incorporate all hourly observations.
- Explore seasonal and holiday-related features.
- Use chronological splitting for time-dependent evaluation.
- Implement feature scaling consistently within model pipelines.
- Compare additional regression algorithms such as Random Forest and Gradient Boosting.
- Evaluate models using MAE, RMSE, and R².
- Apply cross-validation and hyperparameter tuning.
- Develop a Streamlit application for interactive rental demand predictions.

## Author

**Radhika Agarwal**

B.Tech — Computer Science and Engineering

**GitHub:** [Radhika-238](https://github.com/Radhika-238)

---

This project demonstrates practical regression modeling using traditional Machine Learning and neural networks to analyze and predict bicycle rental demand.
