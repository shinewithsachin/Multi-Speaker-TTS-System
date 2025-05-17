import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class SpeakerDataset(Dataset):
    def __init__(self, embeddings_path):
        data = np.load(embeddings_path, allow_pickle=True).item()
        self.speakers = list(data.keys())
        self.data = []

        for speaker, emb_list in data.items():
            if isinstance(emb_list, np.ndarray) and emb_list.shape == (192,):
                
                self.data.append((speaker, emb_list))
            elif isinstance(emb_list, list):
                for emb in emb_list:
                    emb = np.array(emb)
                    if emb.shape == (192,):
                        self.data.append((speaker, emb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        speaker, embedding = self.data[idx]
        embedding = np.array(embedding, dtype=np.float32)
        return torch.tensor(embedding), self.speakers.index(speaker)


class SimpleTTSModel(nn.Module):
    def __init__(self, embedding_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)


def train():
    embeddings_path = "data/speaker_embeddings.npy"
    dataset = SpeakerDataset(embeddings_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleTTSModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0.0
        for x, _ in dataloader:
            print("Batch shape:", x.shape)  
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/tts_model.pth")
    print("✅ Model saved to model/tts_model.pth")


if __name__ == "__main__":
    train()
