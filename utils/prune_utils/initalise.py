import utils.prune_utils.magnitude_threshold_pruning
import utils.prune_utils.magnitude_percentage_pruning


def scheduler(pruning_config):
    
    if pruning_config['schedule_type'] == 'magnitude_threshold':
        schedule = utils.prune_utils.magnitude_threshold_pruning.schedule(pruning_config)
        return schedule
    
    elif pruning_config['schedule_type'] == 'magnitude_percentage':
        schedule = utils.prune_utils.magnitude_percentage_pruning.schedule(pruning_config)
        return schedule
    
