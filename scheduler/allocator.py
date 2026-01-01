import numpy as np
from collections import deque

class DRFAllocator:
    def __init__(self, total_cpu=16.0, total_mem=32.0):
        print("🚀 DRF Allocator BALANCED Mode Initialized!")
        self.total_cpu = total_cpu
        self.total_mem = total_mem
        self.active_tasks = []
        self.delayed_tasks_queue = deque()
        self.utilization_history = []
        self.fairness_history = []
        self.user_allocations = {}
        self.dominant_shares = {}
        self.completed_tasks_count = 0
        self.preemption_count = 0
        self.delayed_tasks_count = 0
        self.retry_success_count = 0
        self.task_counter = 0
        
    def extract_task_values(self, task):
        """Simple value extraction with pandas safety"""
        try:
            if hasattr(task, 'to_dict'):
                task_dict = task.to_dict()
            elif hasattr(task, 'get'):
                task_dict = task
            else:
                task_dict = {'user_id': 'default', 'cpu_required': 1.0, 'memory_required': 2.0, 'priority': 2}
            
            user_id = str(task_dict.get('user_id', 'default'))
            
            cpu_req = self._safe_float(task_dict.get('cpu_required', 1.0))
            mem_req = self._safe_float(task_dict.get('memory_required', 2.0))
            priority = self._safe_int(task_dict.get('priority', 2))
            
            cpu_req = max(0.1, min(self.total_cpu, cpu_req))
            mem_req = max(0.1, min(self.total_mem, mem_req))
            priority = max(1, min(3, priority))
            
            return user_id, cpu_req, mem_req, priority
        except Exception as e:
            print(f"⚠️ Error in extract_task_values: {e}")
            return 'default', 1.0, 2.0, 2
    
    def _safe_float(self, value):
        """Safely convert to float"""
        try:
            if hasattr(value, 'item'):
                return float(value.item())
            return float(value)
        except:
            return 1.0
    
    def _safe_int(self, value):
        """Safely convert to int"""
        try:
            if hasattr(value, 'item'):
                return int(value.item())
            return int(value)
        except:
            return 2
    
    def calculate_dominant_share(self, user_id):
        """Calculate current dominant share for a user"""
        if user_id not in self.user_allocations:
            return 0.0
        cpu_share = self.user_allocations[user_id]['cpu'] / self.total_cpu
        mem_share = self.user_allocations[user_id]['mem'] / self.total_mem
        return max(cpu_share, mem_share)
    
    def calculate_jains_fairness_index(self):
        """Calculate Jain's Fairness Index"""
        if not self.dominant_shares or len(self.dominant_shares) < 2:
            return 1.0
            
        shares = list(self.dominant_shares.values())
        sum_shares = sum(shares)
        sum_squares = sum(s * s for s in shares)
        
        return (sum_shares ** 2) / (len(shares) * sum_squares) if sum_squares > 0 else 1.0

    def get_fairness_report(self):
        """Generate fairness report"""
        fairness = self.calculate_jains_fairness_index()
        
        report = {
            'jains_index': fairness,
            'user_count': len(self.dominant_shares),
            'dominant_shares': self.dominant_shares.copy(),
            'fairness_level': 'EXCELLENT' if fairness >= 0.9 else 'GOOD' if fairness >= 0.7 else 'POOR',
            'completed_tasks': self.completed_tasks_count,
            'preemptions': self.preemption_count,
            'delayed_tasks': self.delayed_tasks_count,
            'queued_tasks': len(self.delayed_tasks_queue),
            'retry_success': self.retry_success_count
        }
        return report
    
    def can_allocate(self, cpu_req, mem_req):
        """Check if resources can be allocated - slightly tighter"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        
        # TIGHTER: 90% instead of 100% - creates occasional contention
        return (used_cpu + cpu_req <= self.total_cpu * 0.9 and 
                used_mem + mem_req <= self.total_mem * 0.9)
    
    def allocate(self, task, predicted_time):
        """BALANCED allocation with slightly stricter fairness"""
        user_id, cpu_req, mem_req, priority = self.extract_task_values(task)
        
        # REJECT obviously impossible tasks
        if cpu_req > self.total_cpu or mem_req > self.total_mem:
            return f"❌ Rejected {user_id} (impossible task)"
        
        current_fairness = self.calculate_jains_fairness_index()
        current_user_share = self.calculate_dominant_share(user_id)
        
        # STEP 1: Check if resources available
        if self.can_allocate(cpu_req, mem_req):
            new_share = max(
                (self.user_allocations.get(user_id, {'cpu':0})['cpu'] + cpu_req) / self.total_cpu,
                (self.user_allocations.get(user_id, {'mem':0})['mem'] + mem_req) / self.total_mem
            )
            
            # BALANCED FAIRNESS RULES - SLIGHTLY STRICTER
            is_large_task = (cpu_req > 6.0 or mem_req > 12.0)  # Lowered thresholds
            
            # Rule 1: Always allow small tasks
            if (cpu_req + mem_req) <= 2.0:
                allow_allocation = True
                reason = "small task"
            
            # Rule 2: Help disadvantaged users
            elif current_user_share < 0.20:  # Stricter: 20% instead of 25%
                allow_allocation = new_share <= 0.60  # Stricter: 60% instead of 65%
                reason = "helping disadvantaged user"
            
            # Rule 3: Large tasks for non-dominant users
            elif is_large_task and current_user_share < 0.35:  # Stricter: 35% instead of 40%
                allow_allocation = True
                reason = "large task for non-dominant user"
            
            # Rule 4: Normal fairness rules
            else:
                allow_allocation = new_share <= 0.50  # Stricter: 50% instead of 55%
                reason = "normal fairness"
            
            if allow_allocation:
                # ALLOCATE
                self.task_counter += 1
                task_dict = {
                    'task_id': self.task_counter,
                    'cpu_required': cpu_req, 
                    'memory_required': mem_req,
                    'priority': priority, 
                    'user_id': user_id,
                    'original_task': task, 
                    'remaining_time': predicted_time,
                    'allocated_round': len(self.utilization_history),
                    'is_large': is_large_task
                }
                self.active_tasks.append(task_dict)
                
                # Update allocations
                if user_id not in self.user_allocations:
                    self.user_allocations[user_id] = {'cpu': 0.0, 'mem': 0.0}
                self.user_allocations[user_id]['cpu'] += cpu_req
                self.user_allocations[user_id]['mem'] += mem_req
                self.dominant_shares[user_id] = new_share
                
                task_type = "LARGE" if is_large_task else "normal"
                return f"✅ Allocated {task_type} to {user_id} (CPU={cpu_req:.1f}, MEM={mem_req:.1f})"
            else:
                # Only delay if absolutely necessary
                self.delayed_tasks_count += 1
                delayed_task = {
                    'task_id': self.task_counter,
                    'task': task, 
                    'predicted_time': predicted_time,
                    'user_id': user_id, 
                    'cpu_req': cpu_req, 
                    'mem_req': mem_req,
                    'priority': priority, 
                    'attempt_count': 1,
                    'is_large': is_large_task
                }
                self.delayed_tasks_queue.append(delayed_task)
                task_type = "LARGE" if is_large_task else "normal"
                return f"⚠️ Delayed {task_type} {user_id} (fairness: {new_share:.1%})"
        
        else:
            # SIMPLE PREEMPTION without pandas issues
            return self.simple_safe_preemption(task, predicted_time, user_id, current_user_share)
    
    def simple_safe_preemption(self, task, predicted_time, user_id, current_user_share):
        """Preemption with slightly more aggressive rules"""
        user_id, cpu_req, mem_req, priority = self.extract_task_values(task)
        is_large_task = (cpu_req > 6.0 or mem_req > 12.0)  # Lowered thresholds
        
        # More aggressive preemption: allow for medium priority too
        if priority >= 2 and not (is_large_task and current_user_share < 0.25):
            self.delayed_tasks_count += 1
            delayed_task = {
                'task_id': self.task_counter,
                'task': task, 
                'predicted_time': predicted_time,
                'user_id': user_id, 
                'cpu_req': cpu_req, 
                'mem_req': mem_req,
                'priority': priority, 
                'attempt_count': 1,
                'is_large': is_large_task
            }
            self.delayed_tasks_queue.append(delayed_task)
            task_type = "LARGE" if is_large_task else "normal"
            return f"⚠️ Delayed {task_type} {user_id} (no preemption for priority {priority})"
        
        # Find candidate using task IDs (no pandas comparison)
        for i, active_task in enumerate(self.active_tasks):
            # More aggressive: preempt same or lower priority
            if active_task['priority'] >= priority:
                # Calculate resources without this specific task (by index)
                temp_cpu = sum(t['cpu_required'] for j, t in enumerate(self.active_tasks) if j != i)
                temp_mem = sum(t['memory_required'] for j, t in enumerate(self.active_tasks) if j != i)
                
                if (temp_cpu + cpu_req <= self.total_cpu * 0.9 and 
                    temp_mem + mem_req <= self.total_mem * 0.9):
                    
                    # Preempt and allocate
                    preempted_user = active_task['user_id']
                    self.remove_task(active_task)
                    self.active_tasks.pop(i)
                    self.preemption_count += 1
                    
                    self.task_counter += 1
                    task_dict = {
                        'task_id': self.task_counter,
                        'cpu_required': cpu_req, 
                        'memory_required': mem_req,
                        'priority': priority, 
                        'user_id': user_id,
                        'original_task': task, 
                        'remaining_time': predicted_time,
                        'allocated_round': len(self.utilization_history),
                        'is_large': is_large_task
                    }
                    self.active_tasks.append(task_dict)
                    
                    if user_id not in self.user_allocations:
                        self.user_allocations[user_id] = {'cpu': 0.0, 'mem': 0.0}
                    self.user_allocations[user_id]['cpu'] += cpu_req
                    self.user_allocations[user_id]['mem'] += mem_req
                    self.dominant_shares[user_id] = self.calculate_dominant_share(user_id)
                    
                    task_type = "LARGE" if is_large_task else "normal"
                    return f"⚡ Preempted {preempted_user} → Allocated {task_type} to {user_id}"
                    break
        
        # If no preemption possible, delay
        self.delayed_tasks_count += 1
        delayed_task = {
            'task_id': self.task_counter,
            'task': task, 
            'predicted_time': predicted_time,
            'user_id': user_id, 
            'cpu_req': cpu_req, 
            'mem_req': mem_req,
            'priority': priority, 
            'attempt_count': 1,
            'is_large': is_large_task
        }
        self.delayed_tasks_queue.append(delayed_task)
        task_type = "LARGE" if is_large_task else "normal"
        return f"⚠️ Delayed {task_type} {user_id} (no preemption possible)"
    
    def remove_task(self, task):
        """Remove task and update allocations"""
        user_id = task['user_id']
        
        if user_id in self.user_allocations:
            self.user_allocations[user_id]['cpu'] -= task['cpu_required']
            self.user_allocations[user_id]['mem'] -= task['memory_required']
            
            if (abs(self.user_allocations[user_id]['cpu']) < 0.001 and 
                abs(self.user_allocations[user_id]['mem']) < 0.001):
                self.dominant_shares.pop(user_id, None)
                self.user_allocations.pop(user_id, None)
            else:
                self.dominant_shares[user_id] = self.calculate_dominant_share(user_id)

    def release_completed_tasks(self):
        """Release completed tasks"""
        completed_tasks = []
        indices_to_remove = []
        
        for i, task in enumerate(self.active_tasks):
            task['remaining_time'] -= 1
            if task['remaining_time'] <= 0:
                completed_tasks.append(task)
                indices_to_remove.append(i)
                self.completed_tasks_count += 1
                self.remove_task(task)
        
        for i in sorted(indices_to_remove, reverse=True):
            self.active_tasks.pop(i)
        
        return completed_tasks

    def retry_delayed_tasks(self):
        """WORKING retry mechanism that actually succeeds"""
        if not self.delayed_tasks_queue:
            return 0
            
        retried_count = 0
        current_fairness = self.calculate_jains_fairness_index()
        
        print(f"   🔄 Retrying {len(self.delayed_tasks_queue)} delayed tasks")
        
        initial_size = len(self.delayed_tasks_queue)
        
        for _ in range(initial_size):
            if not self.delayed_tasks_queue:
                break
                
            delayed_task = self.delayed_tasks_queue.popleft()
            user_id = delayed_task['user_id']
            cpu_req = delayed_task['cpu_req']
            mem_req = delayed_task['mem_req']
            is_large = delayed_task.get('is_large', False)
            current_user_share = self.calculate_dominant_share(user_id)
            
            if self.can_allocate(cpu_req, mem_req):
                new_share = max(
                    (self.user_allocations.get(user_id, {'cpu':0})['cpu'] + cpu_req) / self.total_cpu,
                    (self.user_allocations.get(user_id, {'mem':0})['mem'] + mem_req) / self.total_mem
                )
                
                if (new_share <= 0.70 or
                    current_user_share < 0.30 or
                    current_fairness >= 0.85 or
                    delayed_task['attempt_count'] >= 2):
                    
                    self.task_counter += 1
                    task_dict = {
                        'task_id': self.task_counter,
                        'cpu_required': cpu_req, 
                        'memory_required': mem_req,
                        'priority': delayed_task['priority'], 
                        'user_id': user_id,
                        'original_task': delayed_task['task'], 
                        'remaining_time': delayed_task['predicted_time'],
                        'allocated_round': len(self.utilization_history),
                        'retried': True,
                        'is_large': is_large
                    }
                    self.active_tasks.append(task_dict)
                    
                    if user_id not in self.user_allocations:
                        self.user_allocations[user_id] = {'cpu': 0.0, 'mem': 0.0}
                    self.user_allocations[user_id]['cpu'] += cpu_req
                    self.user_allocations[user_id]['mem'] += mem_req
                    self.dominant_shares[user_id] = new_share
                    
                    retried_count += 1
                    self.retry_success_count += 1
                    task_type = "LARGE" if is_large else "normal"
                    print(f"      ✅ RETRY SUCCESS: {task_type} task for {user_id} (attempt {delayed_task['attempt_count']})")
                    continue
                else:
                    delayed_task['attempt_count'] += 1
                    if delayed_task['attempt_count'] <= 4:
                        self.delayed_tasks_queue.append(delayed_task)
                    else:
                        print(f"      💀 Giving up on {user_id} after 4 attempts")
            else:
                delayed_task['attempt_count'] += 1
                if delayed_task['attempt_count'] <= 4:
                    self.delayed_tasks_queue.append(delayed_task)
                else:
                    print(f"      💀 Giving up on {user_id} after 4 attempts")
        
        return retried_count

    def utilization(self):
        """Calculate utilization"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        return (used_cpu / self.total_cpu) * 100, (used_mem / self.total_mem) * 100

    def get_system_stats(self):
        """Get system stats"""
        cpu_util, mem_util = self.utilization()
        fairness = self.calculate_jains_fairness_index()
        
        return {
            'active_tasks': len(self.active_tasks),
            'cpu_utilization': cpu_util,
            'mem_utilization': mem_util,
            'fairness_index': fairness,
            'completed_tasks': self.completed_tasks_count,
            'preemptions': self.preemption_count,
            'delayed_tasks': self.delayed_tasks_count,
            'queued_tasks': len(self.delayed_tasks_queue),
            'retry_success': self.retry_success_count,
            'active_users': len(self.dominant_shares)
        }