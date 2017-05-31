from __future__ import division
import numpy as np
import scipy as sp
import h5py, os
from sklearn.decomposition import PCA

__authors__ = "Jesse Livezey"

###########
# Options #
###########

_split = {'train': .8, 'valid': .1, 'test': .1} # how to split test, val, train
assert np.allclose(_split['train']+_split['valid']+_split['test'], 1.) # equal within a tolerance

###########
# 
###########

class ECoG():

    """
    ECoG dataset

    Parameters
    ----------
    filename : str
        Filename for data.
    which_set : str
        'train', 'valid', 'test', or 'augment'
    fold : int
        Which fold to use.
    center : bool
        If True, preprocess so that data has zero mean.
    move : float
        Fraction of data to move through for each fold.
    level_classes: bool
        Flag for making classes even over splits or just sampling randomly.
    consonant_prediction: bool
        Flag for just setting y to consonant class.
    vowel_prediction: bool
        Flag for just setting y to vowel class.
    randomize_labels: bool
        Randomly permutes the labels for the examples.
        Meant for control runs.
    frac_train: float
        Percentage of training set to use during training.
    pm_aug_range: int
        Number of of time shifts to use in augmentation.
    """

    def __init__(self, filename, which_set, # get all the classes labelled and split correctly, etc.
                 bands=None, data_types=None,
                 fold=0,
                 seed=20161022, center=True,
                 move = .1, level_classes=True,
                 randomize_labels=False,
                 consonant_prediction=False,
                 vowel_prediction=False,
                 pm_aug_range=0,
                 frac_train=1.,
                 y_labels=57,
                 min_cvs=10,
                 condense=True):

        self.args = locals()
        if bands is None:
            bands = 'high gamma'
        if data_types is None:
            data_types = 'amplitude'

        possible_data_types = ['complex', 'amplitude', 'phase']
        possible_bands = ['alpha', 'theta', 'beta', 'high beta',
                          'gamma', 'high gamma']

        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')

        if not isinstance(data_types, list):
            if ',' in data_types:
                data_types = data_types.replace(', ', ',')
                data_types = data_types.split(',')
            else:
                data_types = [data_types]
        if not  isinstance(bands, list):
            if ',' in bands:
                bands = bands.replace(', ', ',')
                bands = bands.split(',')
            else:
                bands = [bands]

        assert all(d in possible_data_types for d in data_types)
        assert all(b in possible_bands for b in bands)

        assert len(data_types) == len(bands)
        assert len(data_types) == 1

        rng = np.random.RandomState(seed)

        stored = False

        with h5py.File(filename,'r') as f:
            Xs = [f['X{}'.format(b)].value for b in bands]
            orig_shape = Xs[0].shape
            y = f['y'].value.astype(int)

        def split_indices(indices, frac_train, min_cvs): # yes, this function and the following are only used in init()
            """
            Split indices into training/validation/testing groups.
            """
            num_idx = len(indices)
            if (num_idx >= min_cvs) and (num_idx > 3):
                indices = np.array(indices, dtype=int)
                order = rng.permutation(num_idx)

                n_test = max(int(np.round(num_idx*_split['test'])),1)
                n_valid = max(int(np.round(num_idx*_split['valid'])), 1)
                n_train = num_idx-n_valid-n_test
                assert num_idx == n_train + n_valid + n_test

                train_start = int(np.round(fold*num_idx*move))
                train_end = (train_start+n_train) % num_idx
                valid_start = train_end
                valid_end = (valid_start+n_valid) % num_idx
                test_start = valid_end
                test_end = (test_start+n_test) % num_idx

                if train_end > train_start:
                    train_idx = order[train_start:train_end]
                else:
                    train_idx = np.hstack((order[train_start:],order[:train_end]))
                assert train_idx.size == n_train, (train_start, train_end)

                if valid_end > valid_start:
                    valid_idx = order[valid_start:valid_end]
                else:
                    valid_idx = np.hstack((order[valid_start:],order[:valid_end]))
                assert valid_idx.size == n_valid 

                if test_end > test_start:
                    test_idx = order[test_start:test_end]
                else:
                    test_idx = np.hstack((order[test_start:],order[:test_end]))
                assert test_idx.size == n_test

                if frac_train is not None:
                    assert frac_train > 0.
                    assert frac_train <= 1.
                    n_keep = int(np.round(frac_train*len(train_idx)))
                    extra_idx = train_idx[n_keep:]
                    train_idx = train_idx[:n_keep]
                else:
                    extra_idx = []

                return tuple([indices[idx].tolist() for idx in [train_idx, valid_idx, test_idx, extra_idx]])
            else:
                return tuple([[] for _ in range(3)]) + (indices,)

        def check_indices(tr, va, te, ex):
            """
            Check that all indices were included and the training/validation/testing
            splits are independent.
            """
            tr = set(tr)
            va = set(va)
            te = set(te)
            ex = set(ex)
            union = tr | va | te | ex
            max_val = max(union)
            assert len(union)-1 == max_val
            assert len(tr)+len(va)+len(te)+len(ex) == len(union)

        n_examples = Xs[0].shape[0]
        assert all(n_examples == X.shape[0] for X in Xs)
        self.present_cvs = np.zeros(y_labels, dtype=int)
        if level_classes:
            n_classes = y_labels
            class_indices = {}
            for ii in xrange(n_classes):
                class_indices[str(ii)] = np.nonzero(y == ii)[0].tolist()
            total = 0
            for indices in class_indices.values():
                total += len(indices)
            assert total == y.shape[0]
            train_idx = []
            valid_idx = []
            test_idx = []
            extra_idx = []
            for ii, key in enumerate(sorted(class_indices.keys())):
                tr, va, te, ex = split_indices(class_indices[key], frac_train, min_cvs)
                if len(tr) > 0:
                    self.present_cvs[ii] = 1
                train_idx += tr
                valid_idx += va
                test_idx += te
                extra_idx += ex
        else:
            indices = range(n_examples)
            train_idx, valid_idx, test_idx, extra_idx = split_indices(indices, frac_train, min_cvs)

        check_indices(train_idx, valid_idx, test_idx, extra_idx)

        self.indices = (train_idx, valid_idx, test_idx, extra_idx)

        n_classes = y_labels

        if randomize_labels:
            in_idx = np.concatenate((train_idx, valid_idx, test_idx))
            order = rng.permutation(in_idx.shape[0])
            for X in Xs:
                X[in_idx] = X[in_idx][order]

        X_train = [X[train_idx] for X in Xs]
        X_valid = [X[valid_idx] for X in Xs]
        X_test = [X[test_idx] for X in Xs]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        X_train_tmp = []
        X_valid_tmp = []
        X_test_tmp = []
        for ii, (X, dt) in enumerate(zip(X_train, data_types)):
            n_ex = X.shape[0]
            X_pca = X.reshape(n_ex, -1)
            X_train_tmp.append(X_pca)
            n_ex = X_valid[ii].shape[0]
            X_valid_tmp.append(X_valid[ii].reshape(n_ex,-1))
            n_ex = X_test[ii].shape[0]
            X_test_tmp.append(X_test[ii].reshape(n_ex,-1))
        
        X_train = np.hstack(X_train_tmp)
        X_valid = np.hstack(X_valid_tmp)
        X_test = np.hstack(X_test_tmp)

        self.train_mean = X_train.mean(axis=0, keepdims=True)

        if which_set == 'train':
            topo_view = X_train
            y_final = y_train
        elif which_set == 'valid':
            topo_view = X_valid
            y_final = y_valid
        else:
            topo_view = X_test
            y_final = y_test
        if center:
            topo_view = topo_view-self.train_mean

        shape = topo_view.shape
        topo_view = topo_view[:, np.newaxis, np.newaxis, :]

        order = rng.permutation(topo_view.shape[0])
        topo_view = topo_view[order]
        y_final = y_final[order]
        self.y_final = y_final
        
        if condense and not (vowel_prediction or consonant_prediction):
            available_indxs = sorted(set(y_final))
            curr_idx = 0
            y_condensed = np.zeros_like(y_final)
            self.mapping = np.inf * np.ones(y_labels, dtype=int)
            for old_idx in range(max(available_indxs)+1):
                if old_idx in available_indxs:
                    y_condensed[y_final == old_idx] = curr_idx
                    self.mapping[old_idx] = curr_idx
                    curr_idx += 1
            n_classes = curr_idx
            y_final = y_condensed
        elif vowel_prediction:
            assert consonant_prediction == False
            y_final = y_final % 3
        elif consonant_prediction:
            assert vowel_prediction == False
            y_final = (y_final / 3).astype(int)
        n_batch = topo_view.shape[0]
        X = topo_view
        X = X.reshape(n_batch, 1, orig_shape[1], orig_shape[2])
        self.X = np.transpose(X, axes=(0, 1, 3, 2)).astype('float32')
        self.y = y_final
        self.axes = ('batch',0,'time','channels')

    def get_valid_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'valid'
        return ECoG(**args)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        return ECoG(**args)
