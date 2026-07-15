import csv
import networkx as nx
import numpy as np
from rdkit import Chem

folder = "./data_WS/"

"""
The following code will convert the SMILES format into onehot format
"""


def atom_features(atom):
    HYB_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER]
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Sm', 'Tc', 'Gd', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetHybridization(), HYB_list) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    # map over allowable_set with a boolean match for each element
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    # SMILES -> RDKit mol -> graph
    # print(smile)
    mol = Chem.MolFromSmiles(smile)

    # print(type(mol))
    # number of atoms (vertices)
    c_size = mol.GetNumAtoms()

    features = []
    degrees=[]

    for atom in mol.GetAtoms():
        atom_degree = atom.GetDegree()
        degrees.append(atom_degree)

        feature = atom_features(atom)
        features.append(feature)

    features = np.array(features)


    edges = []
    edge_type = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_type.append(bond.GetBondTypeAsDouble())
    # Undirected graph -> directed: each undirected edge (u,v) becomes (u,v) and (v,u)
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    if not edge_index:
        edge_index = []
    else:
        edge_index = np.array(edge_index).transpose(1, 0)

    return c_size, features, edge_index, edge_type, degrees


def load_drug_smile(file):
    """
    :return: drug_dict mapping drug name -> index,
             drug_smile list of SMILES strings in index order
    """
    reader = csv.reader(open(file))
    # next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[1]
        # dedupe by name: dict maps name -> running index
        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    """
    # optional: build smile -> graph (one-hot atom features inside smile_to_graph)
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    """
    return drug_dict, drug_smile


def convert2graph(drug_smile):
    """
    :param drug_smile: list
    :return: smile_graph mapping SMILES string -> graph tuple from smile_to_graph
    """
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    return smile_graph


if __name__ == '__main__':
    # drug_dict, drug_smile = load_drug_smile('./data_WS/drug_SMILES.csv')
    # print(drug_dict)
    # smile = drug_smile[0: 10]
    # smile_graph = convert2graph(smile)
    # a = smile_graph[smile[1]]
    # print(a)
    # print(a[0])
    # print(np.asarray(a[1]).shape)
    # b = np.asarray(a[1])
    # print(b[:, 0])
    # print(b[0])
    # smile_graph = convert2graph(['[Cl-].[Cl-].[223Ra++]', 'O.O.O.[OH-].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[Na+].[Na+].[Fe+3].[Fe+3].[Fe+3].[Fe+3].[Fe+3].OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O'])
    # print(type(smile_graph))
    # print(smile_graph['O.O.O.[OH-].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[Na+].[Na+].[Fe+3].[Fe+3].[Fe+3].[Fe+3].[Fe+3].OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O'])
    # print(type(smile_graph['CS(=O)(=O)OCCCCOS(C)(=O)=O']))
    # print(np.asarray(smile_graph['CS(=O)(=O)OCCCCOS(C)(=O)=O'][1]).shape)

    pass
