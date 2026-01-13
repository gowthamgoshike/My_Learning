import numpy as np

class NeuralNetwork:
    def __init__(self):
        
        # Layer 1 (Hidden): 3 Inputs -> 4 Neurons
       
        self.weights1 = 2 * np.random.random((3, 4)) - 1
        
        # Layer 2 (Output): 4 Inputs (from hidden) -> 1 Output
        
        self.weights2 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        """
        Activation Function: Maps any number to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        The Forward Pass: pushing data from input to output.
        """
        # --- Layer 1 ---
        # Matrix Dot Product: (Input * Weights1)
        layer1_linear = np.dot(inputs, self.weights1)
        # Apply Activation
        layer1_output = self.sigmoid(layer1_linear)
        
        # --- Layer 2 ---
        # The input for Layer 2 is the output of Layer 1
        layer2_linear = np.dot(layer1_output, self.weights2)
        # Apply Activation
        final_output = self.sigmoid(layer2_linear)
        
        return final_output, layer1_output

if __name__ == "__main__":
   
    input_data = np.array([0.5, 0.2, 0.1])
    
    nn = NeuralNetwork()
    prediction, hidden_state = nn.forward(input_data)
    
    print("Neural Network Architecture")
    print(f"Input: {input_data}")
    print(f"Hidden Layer Output (Internal Features): \n{hidden_state}")
    print("-" * 30)
    print(f"Final Prediction: {prediction}")