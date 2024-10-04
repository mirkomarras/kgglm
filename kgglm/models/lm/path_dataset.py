from os.path import join

import pandas as pd
from datasets import Dataset

from kgglm.utils import get_eid_to_name_map, get_rid_to_name_map


class PathDataset:
    def __init__(self, dataset_name: str, base_data_dir: str = "", task: str = None, sample_size: str = None, n_hop: str = None, plain_text_path=False):
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = join(self.base_data_dir, "paths_random_walk")
        self.task = task
        self.sample_size = sample_size
        self.n_hop = n_hop
        # Currently not used, experimental parameter
        self.plain_text_path = plain_text_path
        self.read_single_csv_to_hf_dataset()
        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.dataset_name)
        self.rid2name = get_rid_to_name_map(self.dataset_name)

    def read_csv_as_dataframe(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(join(self.data_dir, filename), header=None, names=["path"], index_col=None)

    def read_single_csv_to_hf_dataset(self) -> None:
        filename = f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt'

        df = self.read_csv_as_dataframe(filename)
        self.dataset = Dataset.from_pandas(df)

    def show_random_examples(self) -> None:
        print(self.dataset["path"][:10])
