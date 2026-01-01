class FIFOAllocator:
    """Simple First-In-First-Out allocator for baseline comparison"""
    
    def __init__(self, total_cpu=16.0, total_mem=32.0):
        self.total_cpu = total_cpu
        self.total_mem = total_mem
        self.active_tasks = []
        self.utilization_history = []
        self.completed_tasks = []
        
    def allocate(self, task, predicted_time):
        used_cpu = sum(t['task']['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['task']['memory_required'] for t in self.active_tasks)
        
        # Simple FIFO - if resources available, allocate
        if (task['cpu_required'] + used_cpu <= self.total_cpu and 
            task['memory_required'] + used_mem <= self.total_mem):
            task_id = id(task)
            self.active_tasks.append({'id': task_id, 'task': task, 'remaining_time': predicted_time})
            return "✅ Allocated"
        else:
            return "⚠️ Delayed"
    
    def release_completed_tasks(self):
        finished = []
        for t in self.active_tasks[:]:
            t['remaining_time'] -= 1
            if t['remaining_time'] <= 0:
                finished.append(t)
                self.active_tasks.remove(t)
                self.completed_tasks.append(t)
        return finished
    
    def utilization(self):
        used_cpu = sum(t['task']['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['task']['memory_required'] for t in self.active_tasks)
        cpu_util = (used_cpu / self.total_cpu) * 100
        mem_util = (used_mem / self.total_mem) * 100
        self.utilization_history.append((cpu_util, mem_util))
        return cpu_util, mem_util