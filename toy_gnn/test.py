import torch
from toy_gnn.model import TOYGNN
from toy_gnn.utils import generate_random_graph, create_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model, test_graphs):
    """Evaluate model performance on test graphs"""
    print("\nEvaluating model on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(test_graphs, desc="Testing graphs"):
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.numpy())
            all_labels.extend(data.y.numpy())
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    accuracy = 100 * correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # Calculate and print metrics
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(all_labels, all_preds, 
                              target_names=['Single Edge', 'Multiple Edges']))
    
    # Create confusion matrix
    print("\nGenerating confusion matrix plot...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Single Edge', 'Multiple Edges'],
                yticklabels=['Single Edge', 'Multiple Edges'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    return all_preds, all_labels

def test_on_new_graph(model):
    """Test model on a new random graph"""
    print("\nTesting on new random graph...")
    test_graph = generate_random_graph(min_nodes=15, max_nodes=20)
    model.eval()
    
    with torch.no_grad():
        out = model(test_graph.x, test_graph.edge_index)
        pred = out.argmax(dim=1)
        probs = torch.nn.functional.softmax(out, dim=1)
        
        print("\nDetailed Results:")
        print("-" * 50)
        print(f"Number of nodes: {test_graph.x.shape[0]}")
        print("\nNode-by-node analysis:")
        for i in range(len(pred)):
            print(f"Node {i}:")
            print(f"  True label: {'Multiple Edges' if test_graph.y[i] == 1 else 'Single Edge'}")
            print(f"  Predicted: {'Multiple Edges' if pred[i] == 1 else 'Single Edge'}")
            print(f"  Confidence: {probs[i].max().item()*100:.2f}%")
        
        accuracy = (pred == test_graph.y).sum().item() / len(pred)
        print(f"\nOverall accuracy: {accuracy*100:.2f}%")

def main():
    # Load trained model
    model = torch.load('best_model.pt')
    _, _, test_graphs = create_dataset(num_graphs=50)
    
    # Evaluate on test set
    evaluate_model(model, test_graphs)
    
    # Test on new random graph
    test_on_new_graph(model)

if __name__ == "__main__":
    main()
