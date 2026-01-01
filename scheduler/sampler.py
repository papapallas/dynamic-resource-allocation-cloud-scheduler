class DecentralizedSampler:
    def __init__(self, sample_size=3):
        self.sample_size = sample_size

    def sample_tasks(self, task_df):
        if task_df.empty:
            return pd.DataFrame()
        # Weighted by priority for fairness
        total_priority = task_df['priority'].sum()
        probabilities = task_df['priority'] / total_priority
        return task_df.sample(n=min(self.sample_size, len(task_df)), weights=probabilities)
