# debug_users.py
import pandas as pd
from scheduler.allocator import DRFAllocator
from scheduler.sampler import DecentralizedSampler
from ml_model.predictor import TaskDurationPredictor

def debug_user_flow():
    print("🔍 Debugging user flow through the system...")
    
    # Load data
    df = pd.read_csv("data/alibaba_style_data.csv")
    print(f"📊 Data has {len(df)} tasks from {df['user_id'].nunique()} users")
    print(f"👥 Users: {df['user_id'].unique()}")
    
    # Test sampler
    sampler = DecentralizedSampler(sample_size=3)
    sampled = sampler.sample_tasks(df.head(10))  # Sample first 10 tasks
    print(f"\n🎲 Sampled {len(sampled)} tasks:")
    for idx, task in sampled.iterrows():
        print(f"   Task {idx}: User {task['user_id']}, CPU={task['cpu_required']}, MEM={task['memory_required']}")
    
    # Test allocator with user tasks
    allocator = DRFAllocator()
    predictor = TaskDurationPredictor()
    
    print(f"\n🧪 Testing allocation with users:")
    for idx, task in sampled.iterrows():
        pred = predictor.predict(task)
        result = allocator.allocate(task, pred)
        print(f"   {task['user_id']}: {result}")

if __name__ == "__main__":
    debug_user_flow()