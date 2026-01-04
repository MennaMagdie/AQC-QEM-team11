import torch
import numpy as np
from model import QuantumErrorMitigator

class QEM_API:
    def __init__(self, weights_path="best_model_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantumErrorMitigator()
        
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("Model loaded successfully.")
        except FileNotFoundError:
            raise Exception(f"Could not find {weights_path}. Make sure it is in the same folder.")
    
        self.model.to(self.device)
        self.model.eval()

    def mitigate(self, noisy_probs, metadata):
        full_input = np.concatenate([metadata, noisy_probs])
        input_tensor = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            log_output = self.model(input_tensor)
            clean_probs = torch.exp(log_output).cpu().numpy()[0]
            
        return clean_probs

if __name__ == "__main__":
    # dummy data for testing
    test_meta = [2, 10, 5, 0.01] 
    
    test_noisy = np.random.rand(16)
    test_noisy /= test_noisy.sum()

    mitigator = QEM_API()
    result = mitigator.mitigate(test_noisy, test_meta)
    
    print("\nTest Successful!")
    print("Input shape:", test_noisy.shape)
    print("Output shape:", result.shape)
    print("Output sum (should be approx 1.0):", result.sum())