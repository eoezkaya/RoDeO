class RegressionTest:
    def __init__(self, name,directory_name, num_trials, optimal_value,target,scale):
        self.name = name
        self.directory_name = directory_name
        self.num_trials = num_trials
        self.optimal_value = optimal_value
        self.scale_factor = scale
        self.target_value = target
