import utils.prune_utils.magnitude_pruning


def scheduler(pruning_config):
    
    if pruning_config['schedule_type'] == 'magnitude':
        schedule = utils.prune_utils.magnitude_pruning.schedule(pruning_config)
        return schedule
    
