from rdkit import Chem
from rdkit.Chem import AllChem

caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(caffeine_smiles)
mol = Chem.AddHs(mol)

params = AllChem.ETKDGv3()
params.randomSeed = 42
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)

energies = []
for cid in conf_ids:
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
    ff.Minimize()
    energies.append((cid, ff.CalcEnergy()))

for cid, e in sorted(energies, key=lambda x: x[1]):
    print(f"Conformer {cid}: MMFF94 energy = {e:.2f} kcal/mol")

writer = Chem.SDWriter("caffeine_conformers.sdf")
for cid in conf_ids:
    writer.write(mol, confId=cid)
writer.close()
print("Saved caffeine_conformers.sdf")
