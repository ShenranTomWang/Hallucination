from abc import ABC, abstractmethod
import json
import pandas as pd
import os

class Dataset(ABC):
    DATA_PATH = None

    def __init__(self) -> None:
        super().__init__()
        if self.DATA_PATH is None:
            raise NotImplementedError("DATA_PATH is not defined")

        if not os.path.exists(self.DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {self.DATA_PATH}")

    
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
        if i < 0 or i >= len(self.data):
            raise IndexError("Index out of bounds")
        return self.data["Question"][i]
    
    def write(self, column: str, i: int, value: any):
        if column not in self.data.columns:
            self.data[column] = None
        self.data.at[i, column] = value
        
    def to_csv(self, filename: str):
        self.data.to_csv(filename)
        
    def len(self) -> int:
        return len(self.data)
    
class QAData(Dataset):
    
    DATA_PATH = "./data/qa_data.json"
    
    def __init__(self) -> None:
        super().__init__()
        with open(self.DATA_PATH, 'r', encoding="utf-8") as file:
            self.data = pd.DataFrame([json.loads(line) for line in file])
        
    def get_question(self, i: int) -> str:
        if i < 0 or i >= len(self.data):
            raise IndexError("Index out of bounds")
        
        knowledge = self.data["knowledge"].iloc[i]
        question = self.data["question"].iloc[i]
        return f"Knowledge: {knowledge}\nQuestion: {question}"
    
    def write(self, column: str, i: int, value: any):
        if i < 0 or i >= len(self.data):
            raise IndexError("Index out of bounds")
        if column not in self.data.columns:
            self.data[column] = None
        self.data.loc[i, column] = value
        
    def to_csv(self, filename: str):
        self.data.to_csv(filename)
        
    def len(self) -> int:
        return len(self.data)

class UMWPDataset(Dataset):
    DATA_PATH = "./data/UMWP.jsonl"

    def __init__(self) -> None:
        super().__init__()
        # with open(self.DATA_PATH, 'r') as file:
        #     self.data = pd.DataFrame([json.loads(line) for line in file])
        with open(self.DATA_PATH, 'r') as file:
            full_data = pd.DataFrame([json.loads(line) for line in file])
            if len(full_data) >= 400:
                self.data = pd.concat([full_data.iloc[:50], full_data.iloc[-50:]]).reset_index(drop=True)
            else:
                self.data = full_data

    def get_question(self, i: int) -> str:
        if 0 <= i < len(self.data):
            return self.data.loc[i, 'question']
        else:
            raise IndexError("Index out of range")

    def write(self, column: str, i: int, value: any):
        if column not in self.data.columns:
            self.data[column] = None
        if 0 <= i < len(self.data):
            self.data.at[i, column] = value
        else:
            raise IndexError("Index out of range")

    def to_csv(self, filename: str):
        self.data.to_csv(filename, index=False)

    def len(self) -> int:
        return len(self.data)


# def test_umwp_dataset():
#     dataset = UMWPDataset()
#     print(dataset.data)
#
# test_umwp_dataset()



