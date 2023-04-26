import logging
import sys
from pathlib import Path
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, DataStructs
import deepchem
import numpy as np
import multiprocessing as mp
from rdkit import RDLogger
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score


RDLogger.DisableLog('rdApp.*')

def setup_logger(logging_dir, logging_file, isstd=True):

    log_path = Path(logging_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path.joinpath(logging_file), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    if isstd:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
    return logger

def valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return False
    if not mol:
        return False
    return True

def remove_invalid_mol(rxn_smiles, log_dict=None):
    '''
    :param rxn_smiles
    :return: rxn_smiles
    remove invalid mol in rxn_smiles
    '''
    if ">>" not in rxn_smiles:
        return None
    reacs, prods = rxn_smiles.split(">>")
    if not reacs or not prods:
        return None
    reacs = reacs.split(".")
    prods = prods.split(".")
    new_reacs = [smi for smi in reacs if valid_smiles(smi)]
    new_prods = [smi for smi in prods if valid_smiles(smi)]
    if len(new_reacs) < len(reacs) or len(new_prods) < len(prods):
        if log_dict:
            log_dict['rxn_cnt_of_remove_invalid_mol'] = log_dict.get('rxn_cnt_of_remove_invalid_mol', 0) + 1
    if not new_reacs or not new_prods:
        return None
    new_reacs = ".".join(new_reacs)
    new_prods = ".".join(new_prods)
    return ">>".join([new_reacs, new_prods])

def remove_invalid_reac(smiles):
    reacs = smiles.split(".")
    new_reacs = [smi for smi in reacs if valid_smiles(smi)]
    return '.'.join(new_reacs)

def get_smiles(smiles):
    smile_list = []
    for smile in smiles.split('|'):
        smile_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
    smiles = '.'.join(smile_list)
    return smiles

def create_Morgan2FP(smi, fpsize=1024, useFeatures=False, useChirality=False):
    smi = get_smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol=mol, radius=2, nBits=fpsize, useFeatures=useFeatures, useChirality=useChirality)
    fp = np.zeros(fpsize, dtype='float32')
    DataStructs.ConvertToNumpyArray(fp_bit, fp)
    return fp.tolist()

def FWeave(smile): # (75,)
    featurizer = deepchem.feat.WeaveFeaturizer()
    features = featurizer.featurize(smile)
    return features[0].get_atom_features().mean(axis=0)

def create_smiles_feature(smiles):
   try:
       mfp = create_Morgan2FP(smiles)
       smiles = get_smiles(smiles)
       smiles = smiles.replace('|', '.').strip().strip('.').strip()
       dfp = FWeave(smiles)
       new_df = mfp + dfp.tolist()
       return new_df
   except:
       return "error"

def create_feature(df, args=None):
    tqdm.pandas(desc='apply')
    # df['newcol'] = pool.map(f, df['col'])
    df['reactants'] = df['rxn'].apply(lambda x: x.split('>>')[0])
    df['products'] = df['rxn'].apply(lambda x: x.split('>>')[1])
    with mp.Pool(mp.cpu_count()//2) as pool:
        # data['rfp'] = data['reactants'].apply(create_smiles_feature)
        # data['pfp'] = data['products'].apply(create_smiles_feature)
        # data['mfp'] = data['smile'].apply(create_smiles_feature)
        df['rfp']=pool.map(create_smiles_feature,df['reactants'])
        df['pfp'] = pool.map(create_smiles_feature,df['products'])
        df['mfp'] = pool.map(create_smiles_feature,df['smile'])
    # data=data=data[data[(len(data['rfp'])>0) & (len(data['pfp'])>0) & (len(data['mfp'])>0)]]
    df=df[(df['rfp']!='error') & (df['pfp']!='error') & (df['mfp']!='error')]
    # df['rdkit_feature'] = df.apply(lambda x: x['rfp'] + x['pfp'] + x['mfp'], axis=1)
    df.loc[:, 'rdkit_feature'] = df.apply(lambda x: x['rfp'] + x['pfp'] + x['mfp'], axis=1)
    feat = np.array(df['rdkit_feature'].tolist())
    target = np.array(df['usage'].tolist())
    return feat, target

def get_RMSE_r2(model, data, args=None):
    y_pred = model.predict(data=data)
    
    test_target = [value for value in data.get_label()]
    prediction = [value for value in y_pred]
    
    rmse_std = np.sqrt(mean_squared_error(test_target, prediction))
    r2_std = r2_score(test_target, prediction)

    return rmse_std, r2_std

if __name__ == '__main__':
    # logger = setup_logger(logging_dir='logs',logging_file='0.log', isstd=False)
    # logger.info(f'test')
    smiles = 'FC1=C(C=CC(=C1)I)N1N=C(C(C(=C1)OC)=O)C1=CC=NN1C1=CC=CC=C1'
    out = create_smiles_feature(smiles)

    print('hello world')