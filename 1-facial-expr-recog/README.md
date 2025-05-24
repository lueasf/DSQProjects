# 1. Facial Expression Recognition with PyTorch 

> We fine-tune a pre-trained EfficientNet-B0 model to classify facial expressions into 7 categories.

This project implements a **Convolutional Neural Network (CNN)** using a pre-trained EfficientNet-B0 model to classify facial expressions into 7 categories: angry, disgust, fear, happy, neutral, sad, and surprise.

**EfficientNet** is a family of neural networks optimized for:
- High accuracy (state-of-the-art performance)
- Low computational cost (fast and lightweight)
It is trained to recognize shapes and patterns to classify general images such as animals, objects, and scenes.

Available in versions B0 (smallest, ideal for prototypes) to B7 (largest, for high-precision tasks), all pre-trained on ImageNet.
-> Implemented via timm (PyTorch Image Models).

## Dataset from Kaggle
https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset


## Data Preprocessing
- Training augmentation : random horizontal flips and rotation to increase robustness.
- Validation transforms : simple conversion to tensor.


## Model Architecture
A CNN is a deep learning architecture specialized for image data. It consists of convolutional layers that extract spatial features, pooling layers that reduce dimensionality, and fully connected layers for classification.
We leverage EfficientNet-B0, a CNN model that scales depth, width, and resolution using a compound scaling method. 

We use :
- **CrossEntropyLoss** as the loss function, suitable for multi-class classification tasks.
- **Adam optimizer** with a learning rate of 0.001, which adapts the learning rate during training.

The final layer outputs a vector of 7 scores (logits), corresponding to the seven emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise.


## Training Process
The training loop is executed for **15 epochs** and includes both training and validation steps. For each epoch:
- **Training phase**:
    The model processes batches of training images.
    - For each batch:
        - The images are passed through the model to compute predictions (logits).

        - The loss is calculated using **CrossEntropyLoss**.

        - Gradients are computed using backpropagation (loss.backward()), and weights are updated with the Adam optimizer (optimizer.step()).

    The model is in training mode (model.train()), allowing layers like dropout and batch normalization to behave accordingly.

- **Validation phase:**
    The model is evaluated on unseen validation images.

    No weight updates are performed; gradients are disabled (torch.no_grad()), and the model is switched to evaluation mode (model.eval()).

    This allows us to monitor generalization performance and detect overfitting.

- **Checkpointing**:
    The model's weights are saved whenever the validation loss improves, ensuring the best version is kept for inference.


## Inference
To make predictions on new images using the trained model:

1 - Load the trained model and its best weights (best-weights.pt).

2 - Preprocess the image: resize and normalize if needed, then convert to a PyTorch tensor and add a batch dimension.

3 - Send the image to the correct device (cuda or cpu) and pass it through the model in evaluation mode.

4 - Apply Softmax to the output logits to get probabilities.

5 - Visualize the image and its predicted emotion distribution using a bar chart for interpretability.
