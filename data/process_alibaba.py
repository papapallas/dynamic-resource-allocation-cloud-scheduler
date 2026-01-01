import pandas as pd
import numpy as np

def process_alibaba_sample():
    """Create realistic sample data based on Alibaba trace patterns"""
    # Based on Alibaba trace characteristics
    np.random.seed(42)
    
    n_tasks = 200  # Larger, more realistic dataset
    
    # Realistic distributions from Alibaba trace analysis
    # Most tasks are small, some are large (bimodal distribution)
    small_tasks = int(n_tasks * 0.7)  # 70% small tasks
    large_tasks = n_tasks - small_tasks
    
    # Small tasks (typical web services, API calls)
    cpu_small = np.random.exponential(0.8, small_tasks)
    mem_small = np.random.exponential(1.5, small_tasks)
    duration_small = np.random.exponential(3.0, small_tasks)
    
    # Large tasks (batch processing, data analytics)
    cpu_large = np.random.uniform(4.0, 12.0, large_tasks)
    mem_large = np.random.uniform(8.0, 24.0, large_tasks) 
    duration_large = np.random.uniform(15.0, 45.0, large_tasks)
    
    # Combine
    cpu_required = np.concatenate([cpu_small, cpu_large])
    memory_required = np.concatenate([mem_small, mem_large])
    task_duration = np.concatenate([duration_small, duration_large])
    
    # Cap at system limits
    cpu_required = np.clip(cpu_required, 0.1, 16.0)
    memory_required = np.clip(memory_required, 0.1, 32.0)
    task_duration = np.clip(task_duration, 0.5, 60.0)
    
    # Priority distribution (1=high, 3=low) - realistic mix
    priority = np.random.choice([1, 2, 3], n_tasks, p=[0.15, 0.60, 0.25])
    
    real_data = pd.DataFrame({
        'task_duration': task_duration,
        'cpu_required': cpu_required,
        'memory_required': memory_required, 
        'priority': priority
    })
    
    # 🚨 CRITICAL FIX: Add multiple users before saving!
    real_data = add_user_distribution(real_data)
    
    # Shuffle the data
    real_data = real_data.sample(frac=1).reset_index(drop=True)
    
    real_data.to_csv("data/alibaba_style_data.csv", index=False)
    print(f"✅ Generated realistic Alibaba-style dataset with {len(real_data)} tasks")
    print(f"👥 Users: {real_data['user_id'].nunique()} unique users")
    print("\n📊 Dataset Statistics:")
    print(real_data.describe())
    
    print(f"\n📈 Resource Distribution:")
    print(f"CPU:  Avg {real_data['cpu_required'].mean():.1f}, Max {real_data['cpu_required'].max():.1f}")
    print(f"Memory: Avg {real_data['memory_required'].mean():.1f}, Max {real_data['memory_required'].max():.1f}")
    print(f"Duration: Avg {real_data['task_duration'].mean():.1f}s, Max {real_data['task_duration'].max():.1f}s")
    
    print(f"\n👥 User Distribution:")
    print(real_data['user_id'].value_counts())
    
    return real_data

def add_user_distribution(data):
    """Add multiple users to simulate multi-tenant environment"""
    # Realistic cloud user distribution: few users submit many tasks
    users = ['user_a', 'user_b', 'user_c', 'user_d']
    # Probability distribution: user_a submits 40% of tasks, user_b 30%, etc.
    data['user_id'] = np.random.choice(users, len(data), p=[0.4, 0.3, 0.2, 0.1])
    return data

def analyze_current_data():
    """Quick function to check your current data"""
    print("\n🔍 Analyzing current alibaba_style_data.csv...")
    try:
        df = pd.read_csv("data/alibaba_style_data.csv")
        print(f"📊 Current data has {len(df)} tasks")
        if 'user_id' in df.columns:
            print(f"👥 Unique users: {df['user_id'].nunique()}")
            print("User distribution:")
            print(df['user_id'].value_counts())
        else:
            print("❌ No user_id column found!")
    except FileNotFoundError:
        print("❌ alibaba_style_data.csv not found!")

if __name__ == "__main__":
    # First, check what we currently have
    analyze_current_data()
    
    print("\n🔄 Generating new data with multiple users...")
    # Generate new data with proper user distribution
    process_alibaba_sample()