from utils import pathjoin

class DataManager:
    """This object will handle merging the task specific training files and processing them into dataloaders"""

    def __init__(self, build_directory, dataloader_directory):
        self.build_directory = build_directory
        self.dataloader_directory = dataloader_directory

if __name__ == "__main__":
    manager = DataManager("data\\build", "data\\dataloader")
