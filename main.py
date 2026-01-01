import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class FixedDRFAllocator:
    def __init__(self, total_cpu=16.0, total_mem=32.0):
        self.total_cpu = total_cpu
        self.total_mem = total_mem
        self.active_tasks = []
        self.user_allocations = {}
        self.user_cumulative_usage = {}
        self.completed_tasks_count = 0
        self.utilization_history = []
        self.fairness_history = []
        self.active_tasks_history = []
        self.preemption_count = 0
        
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
        """Check if resources are available - with buffer"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        
        # Leave 20% buffer to avoid overloading
        return (used_cpu + cpu_req <= self.total_cpu * 0.8 and 
                used_mem + mem_req <= self.total_mem * 0.8)
    
    def try_preemption(self, task):
        """Simple preemption - find one task to preempt to make room"""
        user_id = task['user_id']
        cpu_req = float(task['cpu_required'])
        mem_req = float(task['memory_required'])
        
        # Find a task from a user with higher dominant share
        for active_task in self.active_tasks:
            active_user = active_task['user_id']
            
            # Don't preempt from the same user
            if active_user == user_id:
                continue
                
            current_share = self.calculate_dominant_share(user_id)
            active_share = self.calculate_dominant_share(active_user)
            
            # Preempt if active user has significantly higher share
            if active_share > current_share + 0.2:  # 20% threshold
                print(f"   🔄 PREEMPTING {active_user}'s task to allocate {user_id}")
                self.preemption_count += 1
                
                # Remove the preempted task
                self.active_tasks.remove(active_task)
                
                # Update user allocations for preempted task
                self.user_allocations[active_user]['cpu'] -= active_task['cpu_required']
                self.user_allocations[active_user]['mem'] -= active_task['memory_required']
                
                # Now try to allocate the new task
                if self.can_allocate(cpu_req, mem_req):
                    return self._allocate_task(task, "✅ Allocated (with preemption)")
                else:
                    # If still can't allocate, put preempted task back (simplified)
                    self.active_tasks.append(active_task)
                    self.user_allocations[active_user]['cpu'] += active_task['cpu_required']
                    self.user_allocations[active_user]['mem'] += active_task['memory_required']
                    return "⚠️  Delayed (preemption failed)"
        
        return "⚠️  Delayed (no preemption candidate)"
    
    def _allocate_task(self, task, message):
        """Helper method to allocate a task"""
        user_id = task['user_id']
        cpu_req = float(task['cpu_required'])
        mem_req = float(task['memory_required'])
        
        task_dict = {
            'cpu_required': cpu_req, 
            'memory_required': mem_req,
            'user_id': user_id,
            'remaining_time': 2
        }
        self.active_tasks.append(task_dict)
        
        # Update user allocations
        if user_id not in self.user_allocations:
            self.user_allocations[user_id] = {'cpu': 0.0, 'mem': 0.0}
        self.user_allocations[user_id]['cpu'] += cpu_req
        self.user_allocations[user_id]['mem'] += mem_req
        
        return message
    
    def allocate(self, task):
        """Allocation with preemption support"""
        user_id = task['user_id']
        cpu_req = float(task['cpu_required'])
        mem_req = float(task['memory_required'])
        
        # Check if task fits with buffer
        if self.can_allocate(cpu_req, mem_req):
            current_share = self.calculate_dominant_share(user_id)
            new_share = max(
                (self.user_allocations.get(user_id, {'cpu':0})['cpu'] + cpu_req) / self.total_cpu,
                (self.user_allocations.get(user_id, {'mem':0})['mem'] + mem_req) / self.total_mem
            )
            
            # IMPROVED FAIRNESS: Always allocate if user has low share
            if current_share < 0.4 or new_share <= 0.5:
                return self._allocate_task(task, "✅ Allocated")
            else:
                return "⚠️  Delayed (fairness)"
        else:
            # Try preemption if normal allocation fails
            return self.try_preemption(task)
    
    def release_completed_tasks(self):
        """Release completed tasks"""
        completed_tasks = []
        remaining_tasks = []
        
        for task in self.active_tasks:
            task['remaining_time'] -= 1
            if task['remaining_time'] <= 0:
                completed_tasks.append(task)
                self.completed_tasks_count += 1
                # Free resources (but keep cumulative usage)
                user_id = task['user_id']
                self.user_allocations[user_id]['cpu'] -= task['cpu_required']
                self.user_allocations[user_id]['mem'] -= task['memory_required']
                
                # Remove user if no allocations
                if (self.user_allocations[user_id]['cpu'] <= 0 and 
                    self.user_allocations[user_id]['mem'] <= 0):
                    del self.user_allocations[user_id]
                    
                # FIXED: Update cumulative usage only when tasks complete
                if user_id not in self.user_cumulative_usage:
                    self.user_cumulative_usage[user_id] = {'cpu': 0.0, 'mem': 0.0}
                # Count each completed task as 1 unit of work done
                self.user_cumulative_usage[user_id]['cpu'] += task['cpu_required']
                self.user_cumulative_usage[user_id]['mem'] += task['memory_required']
            else:
                remaining_tasks.append(task)
        
        self.active_tasks = remaining_tasks
        return completed_tasks
    
    def utilization(self):
        """Calculate utilization"""
        used_cpu = sum(t['cpu_required'] for t in self.active_tasks)
        used_mem = sum(t['memory_required'] for t in self.active_tasks)
        return (used_cpu / self.total_cpu) * 100, (used_mem / self.total_mem) * 100

def create_proper_balanced_data():
    """Create data with more size variation to trigger preemptions"""
    np.random.seed(42)
    n_tasks = 40
    
    users = ['user_a', 'user_b', 'user_c', 'user_d']
    
    user_assignments = []
    for i in range(n_tasks):
        user_assignments.append(users[i % len(users)])
    
    # More varied task sizes - some larger ones to trigger preemptions
    data = {
        'task_id': range(n_tasks),
        'user_id': user_assignments,
        'cpu_required': np.clip(np.random.normal(2.5, 2.0, n_tasks), 0.5, 7.0),  # Increased max
        'memory_required': np.clip(np.random.normal(4.0, 3.0, n_tasks), 0.5, 10.0),  # Increased max
        'priority': np.random.choice([1, 2, 3], n_tasks, p=[0.15, 0.70, 0.15]),
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Data distribution: {df['user_id'].value_counts().to_dict()}")
    
    # Show large tasks that might trigger preemption
    large_tasks = df[(df['cpu_required'] > 5) | (df['memory_required'] > 7)]
    if not large_tasks.empty:
        print("📢 Large tasks that may trigger preemption:")
        for _, task in large_tasks.iterrows():
            print(f"   Task {task['task_id']}: {task['user_id']} - CPU={task['cpu_required']:.1f}, MEM={task['memory_required']:.1f}")
    
    return df

def run_fixed_simulation():
    """Run simulation with preemption support"""
    
    tasks_pool = create_proper_balanced_data()
    allocator = FixedDRFAllocator(total_cpu=16.0, total_mem=32.0)
    
    print("🚀 DYNAMIC RESOURCE ALLOCATION ALGORITHM SIMULATION")
    print("=" * 50)
    print(f"👥 Users: {tasks_pool['user_id'].nunique()}")
    print(f"📊 Total tasks: {len(tasks_pool)}")
    print("=" * 50)
    
    user_metrics = {}
    for user in tasks_pool['user_id'].unique():
        user_metrics[user] = {'completed': 0, 'allocated': 0}
    
    max_rounds = 20
    tasks_per_round = 3
    
    high_fairness_rounds = 0
    total_delayed = 0
    
    for round_num in range(max_rounds):
        print(f"\n🎯 ROUND {round_num + 1}/{max_rounds}")
        print("-" * 25)
        
        completed = allocator.release_completed_tasks()
        for completed_task in completed:
            user_id = completed_task['user_id']
            user_metrics[user_id]['completed'] += 1
        
        start_idx = round_num * tasks_per_round
        if start_idx >= len(tasks_pool):
            if len(allocator.active_tasks) == 0:
                break
            else:
                print("   💤 Waiting for tasks to complete...")
                cpu_util, mem_util = allocator.utilization()
                fairness = allocator.calculate_jains_fairness_index()
                allocator.utilization_history.append((cpu_util, mem_util))
                allocator.fairness_history.append(fairness)
                allocator.active_tasks_history.append(len(allocator.active_tasks))
                continue
        
        end_idx = min(start_idx + tasks_per_round, len(tasks_pool))
        current_tasks = []
        for i in range(start_idx, end_idx):
            current_tasks.append(tasks_pool.iloc[i])
        
        delayed_count = 0
        for task in current_tasks:
            result = allocator.allocate(task)
            user_id = task['user_id']
            user_metrics[user_id]['allocated'] += 1
            
            if "Delayed" in result:
                delayed_count += 1
                total_delayed += 1
            
            print(f"   👤 {user_id}: CPU={task['cpu_required']:.1f} MEM={task['memory_required']:.1f} | {result}")
        
        cpu_util, mem_util = allocator.utilization()
        fairness = allocator.calculate_jains_fairness_index()
        
        allocator.utilization_history.append((cpu_util, mem_util))
        allocator.fairness_history.append(fairness)
        allocator.active_tasks_history.append(len(allocator.active_tasks))
        
        if fairness >= 0.9:
            high_fairness_rounds += 1
        
        print(f"📊 CPU: {cpu_util:.1f}% | MEM: {mem_util:.1f}% | Fairness: {fairness:.3f}")
        print(f"   Active: {len(allocator.active_tasks)} | Completed: {allocator.completed_tasks_count} | Delayed: {delayed_count}")
        
        if allocator.user_allocations:
            print(f"   👥 User shares: ", end="")
            for user in allocator.user_allocations:
                share = allocator.calculate_dominant_share(user)
                print(f"{user}={share:.1%} ", end="")
            print()
        
        if allocator.completed_tasks_count >= len(tasks_pool):
            print(f"\n✅ All tasks completed!")
            break
        
        time.sleep(0.1)
    
    while len(allocator.active_tasks) > 0 and max_rounds < 25:
        max_rounds += 1
        completed = allocator.release_completed_tasks()
        for completed_task in completed:
            user_id = completed_task['user_id']
            user_metrics[user_id]['completed'] += 1
        
        if len(allocator.active_tasks) > 0:
            cpu_util, mem_util = allocator.utilization()
            fairness = allocator.calculate_jains_fairness_index()
            allocator.utilization_history.append((cpu_util, mem_util))
            allocator.fairness_history.append(fairness)
            allocator.active_tasks_history.append(len(allocator.active_tasks))
        
        if len(allocator.active_tasks) == 0:
            break
    
    generate_final_report(allocator, user_metrics, high_fairness_rounds, min(max_rounds, 20), total_delayed)

def generate_final_report(allocator, user_metrics, high_fairness_rounds, total_rounds, total_delayed):
    """Generate focused final report"""
    print("\n" + "=" * 50)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 50)
    
    final_fairness = allocator.calculate_jains_fairness_index()
    
    print(f"⚖️  FAIRNESS PERFORMANCE")
    print(f"   Final Jain's Index: {final_fairness:.3f}")
    print(f"   Rounds with ≥0.9 fairness: {high_fairness_rounds}/{total_rounds}")
    print(f"   ✅ Target ≥0.9: {'ACHIEVED' if final_fairness >= 0.9 else 'NOT MET'}")
    
    total_allocated = sum(metrics['allocated'] for metrics in user_metrics.values())
    total_completed = sum(metrics['completed'] for metrics in user_metrics.values())
    
    print(f"\n📈 COMPLETION METRICS")
    print(f"   Tasks Allocated: {total_allocated}")
    print(f"   Tasks Completed: {total_completed}")
    print(f"   Completion Rate: {total_completed/total_allocated:.1%}" if total_allocated > 0 else "N/A")
    print(f"   Total Delays: {total_delayed}")
    print(f"   🔄 Total Preemptions: {allocator.preemption_count}")
    
    print(f"\n👥 USER PERFORMANCE")
    for user, metrics in sorted(user_metrics.items()):
        completed = metrics['completed']
        allocated = metrics['allocated']
        success_rate = completed/allocated if allocated > 0 else 0
        print(f"   {user}: {completed}/{allocated} completed ({success_rate:.1%})")
    
    # FIXED: Calculate proper cumulative usage percentages
    print(f"\n🔍 CUMULATIVE RESOURCE USAGE:")
    total_cpu_used = 0
    total_mem_used = 0
    
    # Calculate total work done by all users
    for user in allocator.user_cumulative_usage:
        total_cpu_used += allocator.user_cumulative_usage[user]['cpu']
        total_mem_used += allocator.user_cumulative_usage[user]['mem']
    
    # Calculate and display percentages based on total work done
    for user in sorted(allocator.user_cumulative_usage.keys()):
        cpu = allocator.user_cumulative_usage[user]['cpu']
        mem = allocator.user_cumulative_usage[user]['mem']
        
        # Calculate percentage of total work done (not capacity!)
        if total_cpu_used > 0:
            cpu_share = (cpu / total_cpu_used) * 100
        else:
            cpu_share = 0
            
        if total_mem_used > 0:
            mem_share = (mem / total_mem_used) * 100
        else:
            mem_share = 0
            
        dominant = max(cpu_share, mem_share)
        print(f"   {user}: CPU={cpu:.1f}({cpu_share:.1f}% of work), MEM={mem:.1f}({mem_share:.1f}% of work) → Dominant={dominant:.1f}%")
    
    print(f"   Total work done: CPU={total_cpu_used:.1f}, MEM={total_mem_used:.1f}")
    print(f"   System capacity: CPU={allocator.total_cpu}, MEM={allocator.total_mem}")
    
    # FIXED: Add this line to generate the plots
    generate_performance_plots(allocator, user_metrics)

def generate_performance_plots(allocator, user_metrics):
    """Generate essential performance plots - FIXED to show all at once"""
    if not allocator.utilization_history:
        print("   No data available for plots")
        return
        
    # Create a single figure with all subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Resource Utilization
    plt.subplot(2, 2, 1)
    rounds = range(len(allocator.utilization_history))
    cpu_util = [u[0] for u in allocator.utilization_history]
    mem_util = [u[1] for u in allocator.utilization_history]
    
    plt.plot(rounds, cpu_util, label='CPU Utilization', linewidth=2, marker='o', markersize=3)
    plt.plot(rounds, mem_util, label='Memory Utilization', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Simulation Round')
    plt.ylabel('Utilization (%)')
    plt.title('Resource Utilization Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Fairness Over Time
    plt.subplot(2, 2, 2)
    if allocator.fairness_history:
        plt.plot(rounds, allocator.fairness_history, marker='o', color='green', linewidth=2, markersize=3)
        plt.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (0.9)')
        plt.xlabel('Round')
        plt.ylabel("Jain's Fairness Index")
        plt.title('Fairness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    # Plot 3: Active Tasks Over Time
    plt.subplot(2, 2, 3)
    if hasattr(allocator, 'active_tasks_history') and allocator.active_tasks_history:
        plt.plot(rounds, allocator.active_tasks_history, marker='o', color='purple', linewidth=2, markersize=3)
        plt.xlabel('Round')
        plt.ylabel('Active Tasks')
        plt.title('Active Tasks Over Time')
        plt.grid(True, alpha=0.3)
        if allocator.active_tasks_history:
            max_tasks = max(allocator.active_tasks_history)
            plt.ylim(0, max_tasks + 1)
    
    # Plot 4: FINAL User Work Distribution
    plt.subplot(2, 2, 4)
    users = list(user_metrics.keys())
    
    # Calculate work distribution (percentage of total work done)
    total_cpu_work = 0
    total_mem_work = 0
    
    for user in users:
        if user in allocator.user_cumulative_usage:
            total_cpu_work += allocator.user_cumulative_usage[user]['cpu']
            total_mem_work += allocator.user_cumulative_usage[user]['mem']
    
    work_shares = []
    for user in users:
        if user in allocator.user_cumulative_usage:
            cpu_work = allocator.user_cumulative_usage[user]['cpu']
            mem_work = allocator.user_cumulative_usage[user]['mem']
            # Use the dominant resource in terms of work percentage
            cpu_share = (cpu_work / total_cpu_work) * 100 if total_cpu_work > 0 else 0
            mem_share = (mem_work / total_mem_work) * 100 if total_mem_work > 0 else 0
            work_shares.append(max(cpu_share, mem_share))
        else:
            work_shares.append(0.0)
    
    colors = ['green' if s <= 30 else 'orange' if s <= 40 else 'red' for s in work_shares]
    bars = plt.bar(users, work_shares, color=colors, alpha=0.7)
    plt.ylabel('Work Share (%)')
    plt.title('Final User Work Distribution\n(Percentage of Total Work Done)')
    
    # Add value labels
    for bar, share in zip(bars, work_shares):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{share:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout and show all plots at once
    plt.tight_layout()
    
    # Show all plots in a single window
    print("   📈 Generating performance plots...")
    plt.show()
    
if __name__ == "__main__":
    run_fixed_simulation()