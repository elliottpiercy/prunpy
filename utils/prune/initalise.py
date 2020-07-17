import utils.prune.rewind_pruning
import utils.prune.magnitude_threshold_pruning
import utils.prune.magnitude_percentage_pruning



def scheduler(pruning_config):
    
    if pruning_config['schedule_type'] == 'magnitude_threshold':
        schedule = utils.prune.magnitude_threshold_pruning.schedule(pruning_config)
    
    elif pruning_config['schedule_type'] == 'magnitude_percentage':
        schedule = utils.prune.magnitude_percentage_pruning.schedule(pruning_config)
    
    elif pruning_config['schedule_type'] == 'rewind_pruning':
        schedule = utils.prune.rewind_pruning.schedule(pruning_config)
        
        
    return schedule
