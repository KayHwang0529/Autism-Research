Exploratory Analysis: 
- Generates a **correlation heatmap** using seaborn to visualize relationships between features
- Notable finding: ASD diagnosis correlates strongly with A5, A6, and A9 questionnaire scores

Model Architecture:
- A 3-layer fully connected neural network with mixed ReLu and Sigmoid activation

Output:
- Correlation heatmap
- Runs predictions on the training set
- Applies a **0.5 decision threshold** to convert probabilities to binary labels
- Reports **binary accuracy** using `tf.keras.metrics.BinaryAccuracy`
