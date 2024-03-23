
class Task:
    """This object will be the base class for all other pretraining tasks"""

    def __init__(self, name):
        self.name = name
    
    def task_step(self, batch):
        pass