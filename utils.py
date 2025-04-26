
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def kmerize(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

def encode_sequence(sequence, k=3):
    kmers = kmerize(sequence.upper(), k)
    return " ".join(kmers)
