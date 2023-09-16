import json
import numpy as np
import pyarrow


class Pulsar:
    # notes: currently ignores _isort/__isort and gets sorted versions

    columns = ['toas', 'stoas', 'toaerrs', 'residuals', 'freqs', 'backend_flags']
    vector_columns = ['Mmat', 'sunssb', 'pos_t']
    tensor_columns = ['planetssb']
    # flags are done separately

    metadata = ['name', 'dm', 'dm', 'dmx', 'pdist', 'pos', 'phi', 'theta']

    def __init__(self):
        pass

    @classmethod
    def read_feather(cls, filename):
        f = pyarrow.feather.read_table(filename)
        self = Pulsar()

        for array in Pulsar.columns:
            if array in f.column_names:
                setattr(self, array, f[array].to_numpy())

        for array in Pulsar.vector_columns:
            cols = [c for c in f.column_names if c.startswith(array)]
            setattr(self, array, np.array([f[col].to_numpy() for col in cols]).swapaxes(0,1).copy())

        for array in Pulsar.tensor_columns:
            rows = sorted(set(['_'.join(c.split('_')[:-1]) for c in f.column_names if c.startswith(array)]))
            cols = [[c for c in f.column_names if c.startswith(row)] for row in rows]
            setattr(self, array,
                    np.array([[f[col].to_numpy() for col in row] for row in cols]).swapaxes(0,2).swapaxes(1,2).copy())

        self.flags = {}
        for array in [c for c in f.column_names if c.startswith('flags_')]:
            self.flags['_'.join(array.split('_')[1:])] = f[array].to_numpy()

        meta = json.loads(f.schema.metadata[b'json'])
        for attr in Pulsar.metadata:
            setattr(self, attr, meta[attr])
        if 'noisedict' in meta:
            setattr(self, 'noisedict', meta['noisedict'])

        return self

    to_list = lambda a: a.tolist() if isinstance(a, np.ndarray) else a

    def save_feather(self, filename, noisedict=None):
        pydict = {array: getattr(self, array) for array in Pulsar.columns}

        pydict.update({f'{array}_{i}': getattr(self, array)[:,i] for array in Pulsar.vector_columns
                                                                 for i in range(getattr(self, array).shape[1])})

        pydict.update({f'{array}_{i}_{j}': getattr(self, array)[:,i,j] for array in Pulsar.tensor_columns
                                                                 for i in range(getattr(self, array).shape[1])
                                                                 for j in range(getattr(self, array).shape[2])})

        pydict.update({f'flags_{flag}': self.flags[flag] for flag in self.flags})

        meta = {attr: Pulsar.to_list(getattr(self, attr)) for attr in Pulsar.metadata}
        if noisedict:
            meta['noisedict'] = {par: val for par, val in noisedict.items() if par.startswith(self.name)}

        pyarrow.feather.write_feather(pyarrow.Table.from_pydict(pydict, metadata={'json': json.dumps(meta)}),
                                      filename)
