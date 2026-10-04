import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.regularizers import l2 as l2_reg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, BinaryAccuracy

def build_mlp(hidden_units, dropout, l2, lr, input_dim=8):
    """
    Builds a Multilayer Perceptron (MLP) for binary classification.
    
    Args:
        hidden_units (list): List of integers, where each integer is the number of units in a hidden layer.
        dropout (float): Dropout rate.
        l2 (float): L2 regularization factor.
        lr (float): Learning rate for Adam optimizer.
        input_dim (int): Number of input features.
    """
    model = Sequential()
    # Input layer
    model.add(Input(shape=(input_dim,)))
    
    for units in hidden_units:
        # Dense layer with ReLU: ReLU introduces non-linearity to learn complex patterns, avoiding vanishing gradient.
        # L2 Regularization penalizes large weights to prevent overfitting.
        model.add(Dense(units, activation='relu', kernel_regularizer=l2_reg(l2)))
        
        # BatchNormalization: Normalizes layer inputs to stabilize and accelerate training.
        model.add(BatchNormalization())
        
        # Dropout: Randomly drops units during training to prevent co-adaptation and overfitting.
        model.add(Dropout(dropout))
        
    # Output layer with sigmoid: Sigmoid squashes the output to a [0, 1] range, representing probability.
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model with Adam (adaptive learning rate) and binary_crossentropy (standard for binary classification).
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(name='accuracy'), AUC(name='auc')]
    )
    
    return model
