

def perceive():
    pass

def learn():
    """
    memory()
    reasoning(): what is it? what might be missing? will it be?
    update:
      model
      planning()
        policy
        value
    """

def reasoning():
    """
    Reason about the relationship among data and to the goal with contexts, from model itself or memory given a fixed belief;
    Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
    
    reward
    observation
    transition
    """
    pass

def planning():
    """
    imagine()
    return()
    """
    pass

def act():
    pass
    
def train():
    """
    eval_interval
      train_interval
        act()
        env.step
        perceive()    
    """
