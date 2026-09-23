
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import sys
sys.path.insert(0, '.')
from utils.datasets import rdmol_to_data

caffeine = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
caffeine = Chem.AddHs(caffeine)
AllChem.EmbedMolecule(caffeine, randomSeed=42)

data = rdmol_to_data(caffeine, smiles='CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
with open('caffeine_test.pkl', 'wb') as f:
    pickle.dump([data], f)
print('Built caffeine_test.pkl')
