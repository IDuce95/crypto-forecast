
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logger_manager import LoggerManager
from cache_manager import CacheManager


@dataclass
class DeepLearningConfig:
    
    def __init__(self, data: pd.DataFrame, config: DeepLearningConfig):
        self.config = config
        self.data = data.copy()
        
        if config.feature_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if config.target_column in numeric_columns:
                numeric_columns.remove(config.target_column)
            self.feature_columns = numeric_columns
        else:
            self.feature_columns = config.feature_columns
            
        self.target_column = config.target_column
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        self.scaled_features = self.feature_scaler.fit_transform(data[self.feature_columns])
        self.scaled_target = self.target_scaler.fit_transform(data[[self.target_column]])
        
        self.sequences, self.targets = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.target_scaler.inverse_transform(scaled_target.reshape(-1, 1)).flatten()


class LSTMModel(nn.Module):
    
    def __init__(self, config: DeepLearningConfig, input_size: int):
        super(GRUModel, self).__init__()
        self.config = config
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_size,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(gru_output_size, config.prediction_horizon)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        
        if self.config.use_attention:
            attended_out, _ = self.attention(gru_out, gru_out, gru_out)
            out = attended_out[:, -1, :]
        else:
            out = gru_out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class TransformerModel(nn.Module):
        pe = torch.zeros(seq_len, model_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                           (-np.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x):
        x = self.input_projection(x)
        
        if x.device != self.pos_encoding.device:
            self.pos_encoding = self.pos_encoding.to(x.device)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        transformer_out = self.transformer(x)
        
        out = transformer_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class DeepLearningTrainer:
        Initialize trainer
        
        Args:
            config: Deep learning configuration
        if self.config.model_type.lower() == "lstm":
            model = LSTMModel(self.config, input_size)
        elif self.config.model_type.lower() == "gru":
            model = GRUModel(self.config, input_size)
        elif self.config.model_type.lower() == "transformer":
            model = TransformerModel(self.config, input_size)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
        return model.to(self.device)
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        self.logger.info(f"Starting {self.config.model_type.upper()} training")
        
        cache_key = f"dl_model_{self.config.model_type}_{len(train_data)}_{hash(str(self.config))}"
        cached_model = self.cache_manager.get_cached_model_metadata(cache_key)
        
        if cached_model:
            self.logger.info(f"Found cached model: {cache_key}")
            return cached_model
        
        train_dataset = CryptoTimeSeriesDataset(train_data, self.config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            val_dataset = CryptoTimeSeriesDataset(val_data, self.config)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        input_size = len(train_dataset.feature_columns)
        model = self.create_model(input_size)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                                   f"Train Loss: {avg_train_loss:.6f}, "
                                   f"Val Loss: {avg_val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                                   f"Train Loss: {avg_train_loss:.6f}")
        
        if val_loader is not None and Path("best_model.pth").exists():
            model.load_state_dict(torch.load("best_model.pth"))
        
        results = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "dataset": train_dataset,
            "config": self.config,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None,
            "epochs_trained": len(train_losses)
        }
        
        self.cache_manager.cache_model_metadata(cache_key, {
            "model_type": self.config.model_type,
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1] if val_losses else None,
            "epochs": len(train_losses),
            "config": self.config.__dict__
        })
        
        self.logger.info(f"Training completed - Final train loss: {train_losses[-1]:.6f}")
        return results
    
    def predict(self, model: nn.Module, data: pd.DataFrame, 
                dataset: CryptoTimeSeriesDataset) -> np.ndarray:
        model.eval()
        
        pred_dataset = CryptoTimeSeriesDataset(data, self.config)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in pred_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        predictions_original_scale = dataset.inverse_transform_target(predictions)
        
        return predictions_original_scale


deep_learning_trainer = DeepLearningTrainer(DeepLearningConfig())
