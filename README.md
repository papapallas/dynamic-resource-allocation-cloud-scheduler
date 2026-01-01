# dynamic-resource-allocation-cloud-scheduler
Advanced DRF-based cloud scheduler with ML integration, pre-emption, and real-time monitoring
Project Overview:
This project implements a Python-based cloud scheduler that uses Dominant Resource Fairness (DRF) to allocate resources across multiple users. The system includes intelligent task pre-emption, real-time monitoring, and machine learning for predicting task durations. The goal is to improve resource utilization while maintaining fairness in multi-tenant cloud environments.
Key Features:
· DRF-based Scheduling: Implements Dominant Resource Fairness for CPU and memory allocation
· Dynamic Pre-emption: Allows for task interruption and reallocation based on fairness metrics
· Task Duration Prediction: Uses linear regression to estimate how long tasks will run
· Real-time Monitoring: Live dashboard showing resource usage and performance metrics
· Performance Analytics: Tracks fairness (using Jain's Index), resource utilization, and completion rates
Performance Results:
· Fairness: Achieved Jain's Fairness Index consistently above 0.9
· Resource Utilization: Maintained 70-80% optimal usage across test scenarios
· Task Completion: 85-95% of tasks were successfully processed
· Pre-emption Success: Over 85% of pre-empted resources were successfully reallocated
System Architecture:
The system is built with modular components:
1. Allocation Engine: Handles resource scheduling using DRF algorithm
2. Prediction Module: Uses ML models to estimate task durations
3. Monitoring Dashboard: Real-time visualization of system performance
4. Analytics Component: Calculates and tracks key performance metrics
docs/architecture.png
Getting Started
Prerequisites:
· Python 3.8 or higher
· pip package manager

Installation:
```bash
git clone https://github.com/papapallas/dynamic-resource-allocation-cloud-scheduler.git
cd dynamic-resource-allocation-cloud-scheduler
pip install -r requirements.txt
```
Running the Scheduler:
```bash
python src/main.py --config config/settings.yaml
```
Project Structure:
```
├── src/                 # Source code
│   ├── allocation/      # DRF scheduling logic
│   ├── prediction/      # ML models for duration prediction
│   ├── monitoring/      # Real-time dashboard
│   └── analytics/       # Performance metrics calculation
├── tests/               # Unit and integration tests
├── config/              # Configuration files
├── data/                # Sample datasets and results
└── docs/                # Documentation and diagrams
```
Technologies Used:
· Python: Core implementation language
· NumPy/Pandas: Data processing and analysis
· Scikit-learn: Machine learning for prediction models
· Matplotlib/Seaborn: Data visualization
· Flask: Web dashboard (if applicable)
· PyTest: Testing framework

Testing
Run the test suite with:
```bash
pytest tests/
```

Future Work
Potential improvements include:
· Support for GPU resource scheduling
· More sophisticated ML models for prediction
· Integration with real cloud APIs (AWS, Azure, GCP)
· Enhanced visualization and reporting features

License
This project is for academic purposes as part of my final year project.
