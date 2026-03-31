🔧 Predictive Maintenance using NASA C-MAPSS Dataset

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using deep learning techniques. It utilizes the NASA C-MAPSS Dataset to build a complete predictive maintenance pipeline, including data preprocessing, model training, evaluation, and decision-making.

📌 Project Overview

Predictive maintenance is crucial in aerospace and industrial systems to prevent unexpected failures. This project:

Performs Exploratory Data Analysis (EDA)
Preprocesses multivariate time-series sensor data
Trains deep learning models (LSTM/CNN)
Evaluates performance using MAE and NASA Scoring
Implements a decision engine for actionable insights
Provides a Streamlit dashboard for visualization
📂 Dataset

The dataset consists of simulated turbofan engine degradation data with:

Multiple engine units
Time-series sensor readings
Different operational conditions (FD001–FD004)

Each dataset varies in complexity:

FD001 & FD003: Simpler operating conditions
FD002 & FD004: Multiple operating conditions and fault modes
⚙️ Methodology
1. Data Preprocessing
Normalization of sensor values
RUL calculation for each engine cycle
Sequence/window generation for time-series modeling
2. Model Development
Deep learning models:
LSTM (for temporal dependencies)
CNN (for feature extraction)
Hyperparameter tuning and early stopping to avoid overfitting
3. Evaluation Metrics
MAE (Mean Absolute Error) – measures prediction accuracy
NASA Score – penalizes late predictions more heavily
4. Decision Engine

Based on predicted RUL:

✅ System Healthy
⚠️ Switch to Backup
🚨 Alert Ground Control
📊 Results Summary
Model generalizes across all datasets (FD001–FD004)
Lower MAE observed in more structured datasets
Higher NASA scores indicate penalties due to late predictions
Decision engine successfully categorizes system health states
📈 Visualizations

The project includes:

Training vs Validation Loss curves
RUL prediction plots
EDA visualizations for sensor behavior

👉 Where to include in report:

EDA graphs → Dataset Description / EDA section
Loss curves → Model Training section
Predictions & evaluation → Results and Discussion section
🚀 How to Run
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to project folder
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python train.py

# Run evaluation
python test_pipeline.py

# Launch dashboard
streamlit run app.py
🧠 Key Learnings
Time-series modeling is essential for degradation prediction
Dataset complexity significantly impacts model performance
NASA scoring is more realistic for safety-critical systems
Decision systems add practical value beyond predictions
🔮 Future Work
Transformer-based time-series models
Reinforcement learning for maintenance decisions
Integration with real-world telemetry data
Uncertainty estimation for predictions
