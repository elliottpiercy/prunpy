import utils.prune_utils.magnitude_pruning


def scheduler(schedule_type, schedule_config):
    
    if schedule_type == 'magnitude':
        schedule = utils.prune_utils.magnitude_pruning.schedule(schedule_config)
        return schedule
    
