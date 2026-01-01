import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now try to import
try:
    from scheduler.allocator import DRFAllocator
    from scheduler.sampler import DecentralizedSampler
    from ml_model.predictor import MLModel
    print("✅ All imports successful!")
    
    # Test that classes can be instantiated
    allocator = DRFAllocator()
    sampler = DecentralizedSampler() 
    model = MLModel()
    print("✅ All classes instantiated successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")