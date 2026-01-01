from scheduler.allocator import DRFAllocator
from scheduler.sampler import DecentralizedSampler
from ml_model.predictor import MLModel
import pandas as pd
import numpy as np

print("\n" + "="*50)
print("MALI SCHEDULER DEMO")
print("="*50)

# 1. Initialize all components
print("\n1. Initializing Components...")
allocator = DRFAllocator()
sampler = DecentralizedSampler()
model = MLModel()

# 2. Test the Scheduler Components
print("\n2. Testing Scheduler Components...")
# Test the allocator with a sample task
sample_task_request = {"cpu": 2.5, "memory": 4.0, "task_id": "task_001"}
allocation_result = allocator.allocate(sample_task_request)
print(f"   Allocation Result: {allocation_result}")

# Test the sampler with a task queue
task_queue = ["task_001", "task_002", "task_003", "task_004"]
sampled_task = sampler.sample_task(task_queue)
print(f"   Sampled Task: {sampled_task}")

# 3. Test the ML Model
print("\n3. Testing ML Model...")
# Load and prepare the synthetic data
data = pd.read_csv("data/synthetic_data.csv")
print(f"   Loaded synthetic data with {len(data)} entries")
    
# Prepare features (X) and target (y)
X = data[["cpu_required", "memory_required", "priority"]]
y = data["task_duration"]
    
# Train the model
model.train(X, y)
    
# Make a prediction
sample_input = np.array([[3.0, 6.0, 2]]) # CPU, Memory, Priority
predicted_duration = model.predict(sample_input)
print(f"   For sample input {sample_input[0]}, predicted duration: {predicted_duration[0]:.2f} minutes")

# 4. Integrated Test
print("\n4. Running Integrated Test...")
# Simulate a task being sampled, then allocated, then its duration predicted
simulated_task = {"cpu": 1.5, "memory": 3.0, "task_id": sampled_task, "priority": 2}
print(f"   Simulating full workflow for task: {simulated_task}")
    
alloc_result = allocator.allocate(simulated_task)
ml_input = np.array([[simulated_task['cpu'], simulated_task['memory'], simulated_task['priority']]])
duration_prediction = model.predict(ml_input)
    
print(f"   Final Prediction: Task {simulated_task['task_id']} will take ~{duration_prediction[0]:.2f} minutes to complete.")

print("\n" + "="*50)
print("DEMO COMPLETE! ??")
print("="*50)
