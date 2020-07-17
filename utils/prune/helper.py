def _validate_layer(layer_ID, valid_ID):
    """
    Check whether a layer is valid for pruning. EG: some schedules are for fully connected/CNN etc
    """
    
    if layer_ID.split('_')[0] == valid_ID:
        return True
    else:
        return False
