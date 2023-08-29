import os
import pickle
import FCD as fcd

from .utils import canonicalize_smiles, getstats, loadmodel



def get_metric(name):
    # Don't use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load training set for novelty
    with open("smiles_data") as f:
        smiles_train = {s for s in f.read().split() if s}

    # load submitted smiles. Only read 10000 smiles
    with open("generated.txt") as f:
        smiles_gen = [s for s in f.read().split() if s][:10000]

    smiles_can = canonicalize_smiles(smiles_gen)
    smiles_valid = [s for s in smiles_can if s is not None]
    smiles_unique = set(smiles_valid)
    smiles_novel = smiles_unique - smiles_train

    validity = len(smiles_valid) / len(smiles_gen)
    uniqueness = len(smiles_unique) / len(smiles_gen)
    novelty = len(smiles_novel) / len(smiles_gen)

    if name == 'validity':
        return validity
    elif name == 'uniqueness':
        return uniqueness
    elif name == 'novelty':
        return novelty
    elif name != 'fcd':
        raise ValueError('Invalid metric: %s' % name)

    # Load precomputed test mean and covariance
    with open("teststats.p", 'rb') as f:
        mean_test, cov_test = pickle.load(f)

    model = loadmodel()
    mean_gen, cov_gen = getstats(smiles_valid, model)

    fcd_value = fcd.calculate_frechet_distance(
        mu1=mean_gen,
        mu2=mean_test,
        sigma1=cov_gen,
        sigma2=cov_test)

    return fcd_value


a = get_metric("fcd")


print("finished")