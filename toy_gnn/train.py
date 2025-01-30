import torch
from toy_gnn.model import TOYGNN
from toy_gnn.utils import create_dataset
from tqdm import tqdm

def train_model():
    print("Initializing training...")
    # Create datasets
    print("Generating datasets...")
    train_graphs, val_graphs, test_graphs = create_dataset(num_graphs=20)
    print(f"Created {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs")
    
    # Initialize model
    model = TOYGNN(
        num_node_features=1,
        num_layers=2,
        hidden_dim=16,
        num_classes=2
    )
    print("\nModel architecture:")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\nStarting training loop...")
    best_val_acc = 0.0
    # Training loop
    model.train()
    for epoch in tqdm(range(100), desc="Training epochs"):
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # Train on each graph
        for i, data in enumerate(train_graphs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Calculate training accuracy
            pred = out.argmax(dim=1)
            correct_train += (pred == data.y).sum().item()
            total_train += data.y.size(0)
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_graphs:
                    out = model(data.x, data.edge_index)
                    val_loss += criterion(out, data.y).item()
                    pred = out.argmax(dim=1)
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
            
            val_acc = 100 * correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model, 'best_model.pt')
                print(f"\n[New best model saved] Val Acc: {val_acc:.2f}%")
            
            print(f'\nEpoch {epoch+1:03d}:')
            print(f'Train Loss: {total_loss/len(train_graphs):.4f}')
            print(f'Train Acc: {100*correct_train/total_train:.2f}%')
            print(f'Val Loss: {val_loss/len(val_graphs):.4f}')
            print(f'Val Acc: {val_acc:.2f}%')
            model.train()
    
    # Save final model
    torch.save(model, 'final_model.pt')
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Final model saved as 'final_model.pt'")
    print("Best model saved as 'best_model.pt'")
    
    return model

if __name__ == "__main__":
    model = train_model()
