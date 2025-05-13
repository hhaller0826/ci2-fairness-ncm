import torch as T
import warnings 
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class ProcessedData:
    def __init__(self, df, assignments, categorical_vars=[], discrete_vars=[], test_size=0.1, batch_size=32):
        self._assignments = assignments
        self.columns = self._get_columns(assignments)
        self.categorical_vars = categorical_vars
        self.discrete_vars = set(discrete_vars).intersection(set(categorical_vars))
        self.batch_size = batch_size

        self.train_df, self.test_df = self._get_train_test_split(df, test_size)

        self.train_dataloader = NCMDataset(self.train_df, assignments).get_dataloader(batch_size=batch_size)
        self.test_dataloader = NCMDataset(self.test_df, assignments).get_dataloader(batch_size=batch_size)

    @property
    def assignments(self):
        return self._assignments
    
    @assignments.setter
    def assignments(self, new_assignments):
        self._assignments = new_assignments
        self.columns = self._get_columns(new_assignments)
        self.train_dataloader = NCMDataset(self.train_df, new_assignments).get_dataloader(batch_size=self.batch_size)
        self.test_dataloader = NCMDataset(self.test_df, new_assignments).get_dataloader(batch_size=self.batch_size)

    def _get_train_test_split(self, df, test_size):
        abbr_df = pd.DataFrame(df, columns=list(self.columns))
        self._print_df = pd.DataFrame()

        dat_train, dat_test = train_test_split(abbr_df, test_size=test_size, random_state=42)
        self.scale = {}
        self.ret_map = {}

        for feat in self.columns:
            encoder = LabelEncoder()
            self._print_df[feat+'_orig'] = dat_train[feat]
            if feat in self.categorical_vars:
                dat_train[feat] = encoder.fit_transform(dat_train[feat])
                dat_test[feat] = encoder.fit_transform(dat_test[feat])

                maxval = abbr_df[feat].nunique()-1
                minval = 0
                self.ret_map[feat] = [encoder.inverse_transform([i]).item() for i in range(maxval+1)]
                
            else:
                maxval = abbr_df[feat].max()
                minval = abbr_df[feat].min()
                
            if feat in self.categorical_vars:
                self.scale[feat] = (lambda x, maxval=maxval, minval=minval: T.round((x*(maxval-minval)) + minval))
            else:
                self.scale[feat] = (lambda x, maxval=maxval, minval=minval: (x*(maxval-minval)) + minval)

            # easiest to use NN with a sigmoid so we need to normalize the values between 0 and 1
            # TODO: currently just hoping the real max & min values are in the dataset. if this is grades but everyone got B's and C's, then my algo will never predict A or D
            dat_train[feat] = dat_train[feat].apply(lambda x: (x-minval)/(maxval-minval))
            dat_test[feat] = dat_test[feat].apply(lambda x: (x-minval)/(maxval-minval))

            self._print_df[feat] = dat_train[feat]

        return dat_train, dat_test
    
    def _get_columns(self, assignments):
        cols = []
        for features in assignments.values():
            cols.extend(features)
        return cols

    def get_assigned_scale(self, assignments=None):
        assignments = assignments if assignments else self.assignments
        return {v: [self.scale[assignments[v][i]] for i in range(len(assignments[v]))] for v in assignments}
    
    def print_df(self, n=5, show_orig=False):
        if show_orig: return self._print_df.head(n)
        return self.test_df.head(n)
    
    def to_cat(self, variable, samples):
        """
        expecting samples = torch.tensor([[sample1],[sample2],...])

        example: my_data.to_cat('A', torch.tensor([[0.0, 1.0],[1.,2.2],[2.2, 0.6]]))
        might return [['African-American', 'Greater than 45'],
                    ['Asian', 'Less than 25'],
                    ['Caucasian', '25 - 45']]
        """
        feature = self.assignments[variable]
        n = range(len(feature))
        return [[self.ret_map[feature[i]][sample[i].int()] for i in n] for sample in samples]
                



class NCMDataset(Dataset):
    def __init__(self, df, assignments):
        self.df = df.reset_index(drop=True)
        self.variables = assignments.keys()
        self.assignments = assignments

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for v in self.variables:
                val = self.df.loc[idx, self.assignments[v]]
                tensor = T.tensor(val, dtype=T.float)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                sample[v] = tensor
            return sample
    
    def get_dataloader(self,batch_size=32,shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    