
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      y_logits = model(X)
      y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)
      loss = loss_fn(y_logits, y)
      train_loss += loss.item()
      train_acc += (y_pred == y).sum().item()/len(y_pred)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> Tuple[float, float]:


    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        test_logits = model(X)
        test_loss += loss_fn(test_logits, y).item()

        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
        test_acc += (test_pred == y).sum().item()/len(test_pred)

      test_loss /= len(dataloader)
      test_acc /= len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device,
          epochs: int,
          optimizer: torch.optim.Optimizer) -> Dict[str, List]:


    result = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model = model,
                                         dataloader=train_dataloader,
                                         loss_fn = loss_fn,
                                         optimizer=optimizer,
                                         device = device)
      test_loss, test_acc = test_step(model = model,
                                         dataloader=test_dataloader,
                                         loss_fn = loss_fn,
                                         optimizer=optimizer,
                                         device = device)

      print(f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            )

      result["train_loss"].append(train_loss)
      result["train_acc"].append(train_acc)
      result["test_loss"].append(test_loss)
      result["test_acc"].append(test_acc)

    return result
