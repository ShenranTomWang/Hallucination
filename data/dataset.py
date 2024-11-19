from abc import ABC, abstractmethod
import pandas as pd

class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_question(self, i: int) -> str:
        """return the ith question in dataset

        Args:
            i (int)

        Returns:
            str: question
        """
        pass
    
    @abstractmethod
    def write(self, column: str, i: int, value: any):
        """write to current dataset. One should create the column if it does not exist

        Args:
            column (str): column to write to
            i (int): entry index in column to write to
            value (any): value to write
        """
        pass
    
    @abstractmethod
    def to_csv(self, filename: str):
        """save current dataset to csv file specified by filename.
            Assume the directory to filename already exist

        Args:
            filename (str): filename
        """
        pass
    
    @abstractmethod
    def len(self) -> int:
        """return the length of current dataset
        
        Returns:
            int: length
        """
        pass
    

class TruthfulQA(Dataset):
    
    DATA_PATH = "./data/truthful_QA.csv"
    
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(self.DATA_PATH)
        
    def get_question(self, i: int) -> str:
        return self.data["Question"][i]
    
    def write(self, column: str, i: int, value: any):
        self.data.loc[i, column] = value
        
    def to_csv(self, filename: str):
        self.data.to_csv(filename)
        
    def len(self) -> int:
        return len(self.data)