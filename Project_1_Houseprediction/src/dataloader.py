import pandas as pd 
import os 

class DataLoader:
    def __init__(self,file_path:str):
        self.file_path=file_path

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not Found: {self.file_path}")
        return pd.read_csv(self.file_path)