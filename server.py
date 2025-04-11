from typing import Dict, List, Optional, Union  # Added Union import
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="GCN Inference API",
    description="API for performing graph convolutional network inference",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x



from pydantic import BaseModel
from typing import List, Optional

class FeatureItem(BaseModel):
    # Categorical fields
    proto: str
    service: str
    state: str
    
    # Numeric fields
    dur: float
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int

class InferenceRequest(BaseModel):
    features: List[FeatureItem]
    ids: Optional[List[int]] = None


# Global variables
label_encoders = None
scaler = None
model = None
device = None
train_feature_names = None
categorical_features = []


def load_artifacts():
    """Load encoders, scaler, and model artifacts"""
    global label_encoders, scaler, model, device, train_feature_names, categorical_features
    
    try:
        # Load encoders and scaler
        with open("encoders.pkl", "rb") as f:
            label_encoders, scaler = pickle.load(f)
        
        # Get feature names and categorical features
        train_feature_names = scaler.feature_names_in_
        categorical_features.extend(label_encoders.keys()) if label_encoders else categorical_features.extend([])
        print("categorical_features", categorical_features )
        
        logger.info(f"Loaded categorical features: {categorical_features}")
        logger.info(f"All feature names: {train_feature_names}")
        
        # Initialize device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = len(train_feature_names)
        model = GCN(in_channels=in_channels, hidden_channels=64, out_channels=2).to(device)
        model.load_state_dict(torch.load("best_gnn_model.pt", map_location=device))
        model.eval()
        
        logger.info("Successfully loaded all artifacts")
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        raise


load_artifacts()


def validate_and_preprocess_features(features: List[Dict[str, Union[float, str]]]) -> np.ndarray:
    """Validate and preprocess input features"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([f.dict() for f in features])
        print("df head2 ", df.head())
        # 1. Check for missing features
        missing_cols = set(train_feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        # 2. Process categorical features
        for col in categorical_features:
            if col not in df.columns:
                continue
                
            # Validate categories
            valid_categories = set(label_encoders[col].classes_)
            present_categories = set(df[col].astype(str).unique())
            invalid_categories = present_categories - valid_categories
            
            if invalid_categories:
                raise ValueError(
                    f"Invalid categories in {col}: {invalid_categories}. "
                    f"Valid categories: {valid_categories}"
                )
            
            # Encode categories
            df[col] = label_encoders[col].transform(df[col].astype(str))
        print("df head2 ", df.head())
        # 3. Ensure correct feature order and convert to numpy
        df = df[train_feature_names]
        return df.values.astype(float)
            

    except Exception as e:
        logger.error(f"Feature validation/preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))




class PredictionResult(BaseModel):
    id: int
    predicted_label: int

class APIResponse(BaseModel):
    predictions: List[PredictionResult]
    message: str  # Changed from List[str] to str

@app.post("/predict", response_model=APIResponse)
async def predict(request: InferenceRequest):
    """Endpoint for making predictions"""
    try:
        print("request", request.features)
        logger.info(f"Received {len(request.features)} samples")  # Add this line
        # Validate and preprocess features
        features_array = validate_and_preprocess_features(request.features)
        
        # Scale features
        X_scaled = scaler.transform(features_array)
        
        # Create graph and predict
        X_tensor = torch.tensor(X_scaled, dtype=torch.float)
        
        # Handle single vs multiple samples differently
        n_samples = X_scaled.shape[0]
        print("n_samples", n_samples)
        
        if n_samples == 1:
            # Special case for single sample - create self-loop only
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            # Original kNN approach for multiple samples
            k = min(5, n_samples-1)  # Ensure k is smaller than n_samples
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
            _, indices = nbrs.kneighbors(X_scaled)

            edge_list = []
            for i in range(indices.shape[0]):
                for j in indices[i][1:]:  # skip self-loop
                    edge_list.append([i, j])
                    edge_list.append([j, i])
            edge_index = torch.tensor(edge_list).T

        data = Data(x=X_tensor, edge_index=edge_index)




        with torch.no_grad():
            data = data.to(device)
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1).cpu().numpy()
        print("predictions", predictions)
        # Prepare results
# Prepare results
        ids = request.ids if request.ids is not None else list(range(len(predictions)))
        return {
            "predictions": [{"id": int(id_), "predicted_label": int(pred)} 
                          for id_, pred in zip(ids, predictions)],
            "message": "Inference complete"  # This is a string, not a list
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        


# if __name__ == "__main__":
#     import uvicorn
#     load_artifacts()
#     uvicorn.run(app, host="0.0.0.0", port=8000)