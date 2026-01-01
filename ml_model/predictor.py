import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

class TaskDurationPredictor:
    def __init__(self, use_real_data=True, model_save_path="models/task_predictor.pkl"):
        self.model = LinearRegression()
        self.feature_names = None
        self.user_mapping = {}
        self.is_trained = False
        self.model_save_path = model_save_path
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Try to load existing model first
        if self._load_model():
            print("✅ Loaded pre-trained ML model")
            return
            
        # Otherwise train new model
        self._train_model(use_real_data)
    
    def _train_model(self, use_real_data):
        """Train the ML model with proper validation"""
        print("🔄 Training new ML model...")
        
        try:
            if use_real_data:
                data = pd.read_csv("data/alibaba_style_data.csv")
                print(f"📊 Training on Alibaba-style dataset ({len(data)} tasks)")
            else:
                data = pd.read_csv("data/synthetic_data.csv")
                print(f"📊 Training on synthetic dataset ({len(data)} tasks)")
        except FileNotFoundError:
            print("❌ No data files found! Creating synthetic data...")
            data = self._create_synthetic_data()
        
        # FIXED: Normalize user IDs in the training data
        data = self._normalize_user_ids(data)
        
        # Prepare features and target
        X, y = self._prepare_features(data)
        
        # FIXED: Split into train and test sets for proper validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=data['user_id'] if 'user_id' in data.columns else None
        )
        
        print(f"📈 Training set: {len(X_train)} samples")
        print(f"🧪 Test set: {len(X_test)} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # FIXED: Comprehensive model evaluation
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        # Save the trained model
        self._save_model()
        
        # Feature importance analysis
        self._analyze_features(X.columns)
    
    def _normalize_user_ids(self, data):
        """FIXED: Normalize user IDs consistently"""
        if 'user_id' in data.columns:
            # Create consistent user mapping
            unique_users = data['user_id'].unique()
            self.user_mapping = {user: idx for idx, user in enumerate(sorted(unique_users))}
            
            # Apply mapping to data
            data = data.copy()
            data['user_numeric'] = data['user_id'].map(self.user_mapping)
            
            print(f"👥 User mapping: {self.user_mapping}")
        
        return data
    
    def _prepare_features(self, data):
        """Prepare features for training"""
        # FIXED: Explicit feature selection
        features = ['cpu_required', 'memory_required', 'priority']
        
        if 'user_numeric' in data.columns:
            features.append('user_numeric')
            print("✅ Using user features in model")
        else:
            print("⚠️  Training without user features")
        
        X = data[features]
        y = data['task_duration']
        
        return X, y
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation with both train and test metrics"""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # R² scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # MAE scores (more interpretable)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n📊 MODEL PERFORMANCE REPORT")
        print("=" * 40)
        print(f"R² Score - Train: {train_r2:.3f}, Test: {test_r2:.3f}")
        print(f"MAE (seconds) - Train: {train_mae:.2f}s, Test: {test_mae:.2f}s")
        
        # Overfitting detection
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.1:
            print(f"⚠️  POTENTIAL OVERFITTING: R² gap = {r2_gap:.3f} (> 0.1)")
        elif r2_gap > 0.05:
            print(f"📝 Moderate generalization gap: R² gap = {r2_gap:.3f}")
        else:
            print(f"✅ Good generalization: R² gap = {r2_gap:.3f}")
        
        # MAE interpretation
        print(f"\n📝 Error Analysis:")
        print(f"   Average prediction error: ±{test_mae:.1f} seconds")
        if test_mae < 1.0:
            print("   ✅ Excellent accuracy (error < 1 second)")
        elif test_mae < 2.0:
            print("   ✅ Good accuracy (error < 2 seconds)")
        elif test_mae < 3.0:
            print("   ⚠️  Moderate accuracy (error < 3 seconds)")
        else:
            print("   ❌ Poor accuracy (error > 3 seconds)")
    
    def _analyze_features(self, feature_names):
        """Analyze feature importance and model coefficients"""
        if hasattr(self.model, 'coef_'):
            print(f"\n🔍 FEATURE ANALYSIS")
            print("=" * 30)
            print(f"Base duration: {self.model.intercept_:.2f}s")
            
            for name, coef in zip(feature_names, self.model.coef_):
                effect = "increases" if coef > 0 else "decreases"
                print(f"   {name}: {coef:+.2f} ({effect} duration)")
    
    def _create_synthetic_data(self):
        """Create comprehensive synthetic data for training"""
        np.random.seed(42)
        n_samples = 1000  # More data for better training
        
        users = ['user_a', 'user_b', 'user_c', 'user_d', 'user_e']
        
        data = {
            'task_duration': np.random.exponential(5.0, n_samples),
            'cpu_required': np.random.exponential(2.0, n_samples),
            'memory_required': np.random.exponential(4.0, n_samples),
            'priority': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2]),
            'user_id': np.random.choice(users, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic relationships
        df['task_duration'] = (
            1.0 + 
            df['cpu_required'] * 1.2 + 
            df['memory_required'] * 0.8 + 
            (4 - df['priority']) * 0.5 +  # Higher priority = shorter duration
            np.random.normal(0, 0.5, n_samples)  # Noise
        )
        
        # Cap values
        df['cpu_required'] = np.clip(df['cpu_required'], 0.1, 8.0)
        df['memory_required'] = np.clip(df['memory_required'], 0.1, 16.0)
        df['task_duration'] = np.clip(df['task_duration'], 0.5, 20.0)
        
        # Save for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/synthetic_data.csv', index=False)
        print("💾 Created synthetic_data.csv with 1000 samples")
        
        return df
    
    def _save_model(self):
        """Save the trained model and metadata"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'user_mapping': self.user_mapping,
            'is_trained': self.is_trained
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {self.model_save_path}")
    
    def _load_model(self):
        """Load a pre-trained model"""
        try:
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.user_mapping = model_data['user_mapping']
            self.is_trained = model_data['is_trained']
            return True
        except (FileNotFoundError, EOFError, KeyError):
            return False
    
    def predict(self, task):
        """FIXED: Safe and consistent prediction with explicit feature handling"""
        if not self.is_trained:
            print("⚠️  Model not trained, using fallback prediction")
            return self._fallback_prediction(task)
        
        try:
            # FIXED: Safe value extraction with bounds
            cpu_req = self._safe_extract_value(task, 'cpu_required', 0.1, 8.0)
            mem_req = self._safe_extract_value(task, 'memory_required', 0.1, 16.0)
            priority = self._safe_extract_value(task, 'priority', 1, 3)
            
            # FIXED: Consistent user ID normalization
            user_numeric = 0  # default
            if 'user_id' in task:
                user_id = task['user_id']
                # Normalize user ID to match training data format
                if user_id.startswith('user_user_'):
                    user_id = user_id.replace('user_user_', 'user_')
                user_numeric = self.user_mapping.get(user_id, 0)
            
            # FIXED: Explicit feature construction based on training
            features = []
            feature_order = []
            
            # Always include core features in consistent order
            features.extend([cpu_req, mem_req, priority])
            feature_order.extend(['cpu_required', 'memory_required', 'priority'])
            
            # Include user feature if it was in training
            if 'user_numeric' in self.feature_names:
                features.append(user_numeric)
                feature_order.append('user_numeric')
            
            # Create DataFrame with explicit feature names
            X = pd.DataFrame([features], columns=feature_order)
            
            # Ensure feature order matches training
            X = X[self.feature_names]
            
            pred = self.model.predict(X)[0]
            return max(0.5, round(pred, 2))  # Reasonable minimum
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}, using fallback")
            return self._fallback_prediction(task)
    
    def _safe_extract_value(self, task, key, min_val, max_val):
        """Safely extract and validate numeric values"""
        try:
            if hasattr(task, 'get'):
                value = task.get(key, min_val)
            else:
                value = getattr(task, key, min_val)
            
            # Handle pandas Series
            if hasattr(value, 'item'):
                value = value.item()
            
            value = float(value)
            return max(min_val, min(max_val, value))
        except (ValueError, TypeError, AttributeError):
            return min_val
    
    def _fallback_prediction(self, task):
        """Fallback prediction when model is unavailable"""
        cpu_req = self._safe_extract_value(task, 'cpu_required', 0.1, 8.0)
        mem_req = self._safe_extract_value(task, 'memory_required', 0.1, 16.0)
        
        # Simple heuristic based on resources
        duration = 1.0 + (cpu_req * 1.5) + (mem_req * 0.8)
        return max(1.0, round(duration, 1))
    
    def get_model_info(self):
        """Get model information for debugging"""
        if not self.is_trained:
            return "Model not trained"
        
        info = {
            'features': self.feature_names,
            'user_mapping': self.user_mapping,
            'coefficients': dict(zip(self.feature_names, self.model.coef_)) if hasattr(self.model, 'coef_') else {},
            'intercept': self.model.intercept_ if hasattr(self.model, 'intercept_') else 0
        }
        return info

# Enhanced test function
def test_predictor():
    """Comprehensive predictor testing"""
    print("\n🧪 COMPREHENSIVE PREDICTOR TESTING")
    print("=" * 50)
    
    # Test with different scenarios
    predictor = TaskDurationPredictor(use_real_data=False)
    
    # Test cases covering various scenarios
    test_tasks = [
        {'cpu_required': 2.0, 'memory_required': 4.0, 'priority': 2, 'user_id': 'user_a'},
        {'cpu_required': 8.0, 'memory_required': 16.0, 'priority': 1, 'user_id': 'user_b'},
        {'cpu_required': 0.5, 'memory_required': 1.0, 'priority': 3, 'user_id': 'user_c'},
        {'cpu_required': 4.0, 'memory_required': 8.0, 'priority': 2, 'user_id': 'user_user_a'},  # Test normalization
        {'cpu_required': 1.0, 'memory_required': 2.0, 'priority': 2},  # No user_id
    ]
    
    print("\n📋 PREDICTION RESULTS:")
    for i, task in enumerate(test_tasks, 1):
        pred = predictor.predict(task)
        user = task.get('user_id', 'unknown')
        print(f"   Task {i} (User {user}):")
        print(f"      CPU={task['cpu_required']:.1f}, MEM={task['memory_required']:.1f}, PRIO={task['priority']}")
        print(f"      → Predicted: {pred:.1f}s")
    
    # Show model info
    print(f"\n🔧 MODEL INFO:")
    info = predictor.get_model_info()
    print(f"   Features: {info['features']}")
    print(f"   Users mapped: {len(info['user_mapping'])}")
    print(f"   Intercept: {info['intercept']:.2f}s")

if __name__ == "__main__":
    test_predictor()