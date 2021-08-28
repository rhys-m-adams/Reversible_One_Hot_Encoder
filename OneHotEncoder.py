import pandas
import numpy as np

class OHE:
    def __init__(self, df, ref=[]):
        assert isinstance(df, pandas.DataFrame), 'Input needs to be pandas.DataFrame'
        assert isinstance(ref, list), 'ref needs to be a list'
        forward_map = {}
        df_k = df.keys()
        empty_ref = len(ref)==0
        count = int(0)
        for ii, k in enumerate(df_k):
            curr = list(set(df[k]))
            if empty_ref:
                ref.append(curr[0])

            for elem in curr:
                if ref[ii] != elem:
                    forward_map[(k,elem)] = count
                    count = int(count+1)

        self.forward_map = forward_map
        self.reverse_map = {v:k for k,v in zip(forward_map.keys(), forward_map.values())}
        self.total_count = count
        self.ref = ref
        self.keys = df_k
        self.reverse_keys = {k:ii for ii, k in enumerate(df_k)}

    def transform(self, df):
        assert isinstance(df, pandas.DataFrame), 'Input needs to be pandas.DataFrame'

        out = np.zeros((len(df), self.total_count))
        df_k = df.keys()
        for k in df_k:
            curr = df[k].values
            for ii, val in enumerate(curr):
                ind = self.forward_map.get((k,val), None)
                if not(ind is None):
                    out[ii, int(ind)] = 1
        return out

    def rev_transform(self,np_array):
        assert isinstance(np_array, np.ndarray), 'Input needs to be numpy array'

        out = []
        for row in np_array:
            curr = self.ref.copy()
            inds = np.where(row)[0]
            for ind in inds:
                curr_tuple = self.reverse_map[ind]
                curr[self.reverse_keys[curr_tuple[0]]] = curr_tuple[1]
            out.append(curr)

        return pandas.DataFrame(np.array(out), columns = self.keys)
