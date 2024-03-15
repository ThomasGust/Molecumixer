import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import PIL
import time
import rdkit
import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl
import time

def get_maccs_keys(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    maccs = np.frombuffer(MACCSkeys.GenMACCSKeys(mol).ToBitString().encode(), 'u1') - ord('0')
    return maccs

def get_rdkfp(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    rdkfp = np.frombuffer(Chem.RDKFingerprint(mol).ToBitString().encode(), 'u1') - ord('0')
    return rdkfp
# LAST SUCCESFUL COMPOUND: 133469
if __name__ == "__main__":
    #mol = Chem.MolFromSmiles("CO")
    #print(type(mol))
    #new_generate()
    #generate()
    #mol = Chem.MolFromSmiles("COC(=O)[C@]1(C)CCC[C@@]2(C)C1CC[C@]13C=C(C(C)C)[C@H](CC21)CC3(Cl)C#N")
    #mol = Chem.MolFromSmiles("CO")
    #img.save("test.png")
    #print("FINISHED NOW")
    compounds = pd.read_csv("data\\raw\\chembl_compounds.csv", on_bad_lines='warn', sep=';')
    culled = compounds[['ChEMBL ID', 'Smiles']].dropna()

    x = []
    ls = len(list(culled['Smiles']))
    for i, smiles in enumerate(list(culled['Smiles'])):
        try:
            smiles = Chem.MolFromSmiles(smiles)
            maccs = np.frombuffer(MACCSkeys.GenMACCSKeys(smiles).ToBitString().encode(), 'u1') - ord('0')
            x.append(list(maccs))

            if i % 10_000 == 0:
                print(f"{i}/{ls}: {i/ls*100}")
        except Exception as e:
            print(e, i)

    v = 0

    st = time.time()
    k_100 = KMeans(n_clusters=100, random_state=0, verbose=v).fit(x)
    et = time.time()
    print("FINISHED WITH K=100 CLUSTERING")
    print(et-st)

    st1 = time.time()
    k_500 = KMeans(n_clusters=500, random_state=0, verbose=v).fit(x)
    et1 = time.time()
    print("FINISHED WITH K=500 CLUSTERING")
    print(et1-st1)
    print((et1-st1)/(et-st))

    st2 = time.time()
    k_1000 = KMeans(n_clusters=1_000, random_state=0, verbose=v).fit(x)
    et2 = time.time()
    print("FINISHED WITH K=1000 CLUSTERING")
    print(et2-st2)
    print((et2-st2)/(et1-st1))

    st3 = time.time()
    k_5000 = KMeans(n_clusters=5_000, random_state=0, verbose=v).fit(x)
    et3 = time.time()
    print("FINISHED WITH K=5000 CLUSTERING")
    print(et3-st3)
    print((et3-st3)/(et2-st2))

    st4 = time.time()
    k_10000 = KMeans(n_clusters=10_000, random_state=0, verbose=v).fit(x)
    et4 = time.time()
    print("FINISHED WITH K=10000 CLUSTERING")
    print(et4-st4)
    print((et4-st4)/(et3-st3))

    with open("models.pkl", "wb") as f:
        #pkl.dump([k_100, k_500, k_1000, k_5000, k_10000],f)
        pkl.dump([k_100, k_500, k_1000, k_5000, k_10000],f)
    print("DONE")
