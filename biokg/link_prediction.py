import numpy as np
import pandas as pd
import sklearn
import torch as th
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from biokg.embeddings.base import EmbeddingModel

DEFAULT_CLF = Pipeline([
    ("scaler", StandardScaler()),
    ('pca', PCA(n_components=128)),
    ("model", RandomForestClassifier(n_estimators=100, n_jobs=-1))
])

class LinkPrediction:
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 downstream_model: sklearn.base.BaseEstimator = DEFAULT_CLF):
        self.downstream_model = downstream_model
        self.embedding_model = embedding_model
        self.data = self.embedding_model.data # TODO(fin): Surely there is a better way

    @th.no_grad()
    def train_downstream_model(self) -> sklearn.base.BaseEstimator:
        self.embedding_model.model.eval()
        z = self.embedding_model()
        X = np.stack([th.mul(z[i], z[j]).detach().cpu().numpy() for i, j in self.data.treatment_edge_index.T])
        self.clf = sklearn.base.clone(self.downstream_model)
        y = self.data.y.cpu().numpy()
        self.clf.fit(X, y) 
        return self.clf

    @th.no_grad()
    def eval_downstream_model(self) -> pd.DataFrame:
        self.embedding_model.model.eval()
        z = self.embedding_model()
        X = np.stack([th.mul(z[i], z[j]).detach().cpu().numpy() for i, j in self.data.treatment_edge_index.T])
        y = self.data.y.cpu().numpy()
        predictions = []
        for split_ix in range(len(self.data.train_mask)):
            train_mask = self.data.train_mask.cpu().numpy()[split_ix]
            clf = sklearn.base.clone(self.downstream_model)
            clf.fit(X[train_mask], y[train_mask])
            y_prob = clf.predict_proba(X[~train_mask])[:,1]
            predictions.append(dict(cv_cycle=split_ix, 
                                split="split",
                                y_pred=y_prob>0.5,
                                y_prob = y_prob,
                                y=y[~train_mask]))
        predictions = pd.DataFrame(predictions).explode(["y_pred", "y_prob", "y"])
        return predictions

    @th.no_grad()
    def predict_downstream_model(self, filename:str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not filename:
            filename = self.embedding_model.__class__.__name__
        z = self.embedding_model()

        disease_mask = np.array([i=="Disease" for i in self.embedding_model.node_type])
        molecule_mask = np.array([i=="Molecule" for i in self.embedding_model.node_type])
        
        z_mol = z[molecule_mask]
        z_disease = z[disease_mask]
        
        # Make memory mapped arrays to hold the predictions
        shape =(sum(molecule_mask), sum(disease_mask))
        
        predictions = np.memmap(f"../data/{filename}.predictions.dat", dtype='float16', mode='w+', shape=shape)
        predictions_z_score_disease = np.memmap(f"../data/{filename}.z_score.disease.dat", dtype='float16', mode='w+', shape=shape)
        predictions_z_score_molecule = np.memmap(f"../data/{filename}.z_score.molecule.dat", dtype='float16', mode='w+', shape=shape)
        
        for mol_ix, _z_mol in tqdm(enumerate(z_mol), total=len(z_mol)):
            X = (_z_mol*z_disease).detach().cpu().numpy()
            y_prob = self.clf.predict_proba(X)[:,1]  
            predictions[mol_ix, :] = y_prob
        
        # To prevent frequent fliers, compute the disease-wise and molecule-wise Z-scores.
        for molecule_idx in np.arange(len(predictions_z_score_molecule)):
            query_mol_all_dis_predictions = predictions[molecule_idx, :]
            z_score_mol = (query_mol_all_dis_predictions - np.mean(query_mol_all_dis_predictions)) / np.std(query_mol_all_dis_predictions)
            predictions_z_score_molecule[molecule_idx, :] = z_score_mol
        
        for disease_idx in np.arange(predictions_z_score_disease.shape[1]):
            all_mol_query_dis_predictions = predictions[:, disease_idx]
            z_score_dis = (all_mol_query_dis_predictions - np.mean(all_mol_query_dis_predictions)) / np.std(all_mol_query_dis_predictions)
            predictions_z_score_disease[:, disease_idx] = z_score_dis

        return predictions, predictions_z_score_molecule, predictions_z_score_disease
                    