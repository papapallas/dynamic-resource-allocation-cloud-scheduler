import numpy as np
from collections import defaultdict

class BasicDRFAllocator:
    """Simplified DRF allocator without complex features"""
    
    def __init__(self, total_cpu=16.0, total_mem=32.0):
        self.total_cpu = total_cpu
        self.total_mem = total_mem
        self.active_tasks = []
        self.user_allocations = defaultdict(lambda: {'cpu': 0.0, 'mem': 0.0})
        self.completed_tasks_count = 0
        self.utilization_history = []
        self.fairness_history = []
        
    def calculate_dominant_share(self, user_id):
        """Calculate user's dominant resource share"""
        if user_id not in self.user_allocations:
            return 0.0
        cpu_share = self.user_allocations[user_id]['cpu'] / self.total_cpu
        mem_share = self.user_allocations[user_id]['mem'] / self.total_mem
        return max(cpu_share, mem_share)
    
    def calculate_jains_fairness_index(self):
        """Calculate Jain's Fairness Index"""
        if not self.user_allocations or len(self.user_allocations) < 2:
            return 1.0
            
        shares = [self.calculate_dominant_share(user) for user in self.user_allocations]
        sum_shares = sum(shares)
        sum_squares = sum(s * s for s in shares)
        
        return (sum_shares ** 2) / (len(shares) * sum_squares) if sum_squares > 0 else 1.0
    
    def can_allocate(self, cpu_req, mem_req):
        """Check if resources are available"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        return (used_cpu + cpu_req <= self.total_cpu and 
                used_mem + mem_req <= self.total_mem)
    
    def allocate(self, task):
        """Basic DRF allocation - always allocate if resources available"""
        user_id = task['user_id']
        cpu_req = float(task['cpu_required'])
        mem_req = float(task['memory_required'])
        
        # Check if task fits
        if not self.can_allocate(cpu_req, mem_req):
            return "Delayed (no resources)"
        
        # Simple allocation - always allocate if resources available
        task_dict = {
            'cpu_required': cpu_req, 
            'memory_required': mem_req,
            'user_id': user_id,
            'remaining_time': max(1, int(cpu_req + mem_req))  # Simple duration
        }
        self.active_tasks.append(task_dict)
        
        # Update user allocations
        self.user_allocations[user_id]['cpu'] += cpu_req
        self.user_allocations[user_id]['mem'] += mem_req
        
        return "Allocated"
    
    def release_completed_tasks(self):
        """Release completed tasks"""
        completed_tasks = []
        remaining_tasks = []
        
        for task in self.active_tasks:
            task['remaining_time'] -= 1
            if task['remaining_time'] <= 0:
                completed_tasks.append(task)
                self.completed_tasks_count += 1
                # Free resources
                user_id = task['user_id']
                self.user_allocations[user_id]['cpu'] -= task['cpu_required']
                self.user_allocations[user_id]['mem'] -= task['memory_required']
            else:
                remaining_tasks.append(task)
        
        self.active_tasks = remaining_tasks
        return completed_tasks
    
    def utilization(self):
        """Calculate utilization"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        return (used_cpu / self.total_cpu) * 100, (used_mem / self.total_mem) * 100
    
    def get_stats(self):
        """Get basic statistics"""
        cpu_util, mem_util = self.utilization()
        fairness = self.calculate_jains_fairness_index()
        
        return {
            'active_tasks': len(self.active_tasks),
            'cpu_utilization': cpu_util,
            'mem_utilization': mem_util,
            'fairness_index': fairness,
            'completed_tasks': self.completed_tasks_count
        }