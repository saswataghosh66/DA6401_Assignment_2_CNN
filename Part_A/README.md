# **Aim**:Building a Convolutional Neural Network (CNN) for Image Classification on the iNaturalist Dataset
The primary objective of this project is to design and train a Convolutional Neural Network (CNN) to classify images from the [iNaturalist](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) dataset. This dataset consists of images from various biological categories such as animals, plants, and fungi. The CNN will be trained to recognize and classify these species based on their visual features.

# **Key Steps to Achieve the Aim:**
[The code is organized and explained step-by-step in the Jupyter Notebook. Each step is numbered for clarity. In this README file, I have described the implementation by referring to the corresponding step numbers from the notebook. This makes it easier to follow the logic and match the explanation with the actual code.]

**Step 1:**
 - Importing necessary libraries for deep learning, data handling, preprocessing, model training, evaluation, visualization, and experiment tracking.


**Step 2:**

Set device to GPU if available, else use CPU

Define a helper function to calculate output image size after a convolution

 Define a customizable convolutional neural network builder function 
 
 `get_convnet`:

 - Allows setting number of filters, filter sizes, activation functions, pooling, padding, etc.
 - Supports batch normalization and dropout
 - Automatically computes final fully connected input size based on conv layers
 - Returns the model as an nn.Sequential module
 Example usage: builds and prints a sample CNN model with default parameters

**Step 3:**

- Define model_training() function:
- Moves model to selected device (GPU/CPU)
- Initializes loss function (CrossEntropy) and optimizer (Adam)
- Sets up Weights & Biases (wandb) for logging if enabled
- Trains model over specified number of epochs:
  - Computes training loss and accuracy
  - Evaluates model on validation data each epoch
  - Logs metrics to wandb and prints progress
- After training, displays a plot of training vs. validation accuracy

**Step 4:**

- Visualizes 10 random images from a randomly selected class in the training dataset
- Loads class folders from the training directory
- Picks one random class and samples 10 images from it
- Displays the selected images using matplotlib without axis labels
- Titles each image with the class name for reference

**step 5:**

- Loads image dataset from the given directory and applies transformations
- Applies data augmentation if 'Yes' is passed as the data_augmentation flag
- Uses torchvision transforms for resizing, normalization, and optional augmentation (crop, flip, color jitter, rotation)
- Splits the dataset into training and validation sets (80-20 split)
- Returns DataLoaders for training and validation, the full dataset, and index lists for train/val samples

**Step 6:**

- Calls the data_load function with data augmentation enabled to load training and validation data
- Prints total number of images in the full dataset
- Prints the number of images used for training and validation
- Prints the number of batches in training and validation DataLoaders

**Step 7:**

Wandb Sweep
- Defines a hyperparameter sweep configuration using Bayesian optimization with Weights & Biases (wandb)
- Objective is to maximize validation accuracy ('val_accuracy')
- Specifies the parameter search space:
  - kernel_size: different lists of convolution filter sizes
  - dropout: various dropout probabilities to regularize the model
  - activation: activation functions to try (ReLU, SiLU, Mish, GELU)
  - num_dense: size of the fully connected layer
  - batch_norm: toggle for using batch normalization
  - filter_org: different configurations for the number of filters per convolutional layer
- Initializes the sweep and gets the sweep ID

- Main function to train the model using W&B sweep configuration
- Initializes the model based on sweep parameters, including kernel size, activation function, dropout rate, and more
- Constructs a run name using the current configuration to track experiments in W&B
- Initializes the selected activation function based on sweep config
- Calls `get_convnet()` to build the convolutional model with specified parameters
- Checks that the `train_loader` is defined before training
- Uses the W&B-specified learning rate and calls `model_training()` to train the model
- Finalizes the W&B run with `wandb.finish()`

**Step 8:**

- Loads validation images from the specified directory (`test_data_dir`)
- Randomly selects a class from the validation data folders
- Samples 10 images from the randomly selected class
- Displays the 10 selected images in a 1x10 grid without axis labels
- Titles each image with the class name

**Step 9:**

- Defines the function `test_data_load` to load and preprocess the test dataset
- Applies a series of transformations (resizing, normalization, and optional data augmentation)
- Augmentation includes random resized crop, horizontal flip, color jitter, and random rotation
- Returns a DataLoader for the test dataset to batch the images for evaluation
- Calls the function with `data_augmentation` set to "No"
- Prints the total number of images in the test dataset and the number of batches in the test DataLoader

**Step 10:**

- Defines a function `test` to evaluate the model's performance on the test dataset
- Switches the model to evaluation mode using `model.eval()`
- Initializes counters for correct predictions and total samples
- Loops over the test data and performs predictions without calculating gradients (`torch.no_grad()`)
- Calculates the number of correct predictions and the total number of samples
- Computes the test accuracy as the percentage of correct predictions
- Prints and returns the test accuracy

**Step 11:**

  - Defines the `model_training` function to train and validate the model for a specified number of epochs
- Moves the model to the specified device (GPU or CPU)
- Initializes the loss function (CrossEntropyLoss) and optimizer (Adam) for training
- Tracks training accuracy and validation accuracy for each epoch
- In each epoch:
  - Switches to training mode with `model.train()`
  - Computes the loss and performs backpropagation to update the model's parameters
  - Tracks the training loss and accuracy for the current epoch
  - Switches to evaluation mode with `model.eval()` for validation
  - Computes validation accuracy without gradient calculations using `torch.no_grad()`
  - Prints the training loss, training accuracy, and validation accuracy for each epoch
- Returns the trained model after all epochs are completed

**step 12:**

  - Sets the device to 'cuda' if a GPU is available, otherwise defaults to 'cpu'
- Initializes a convolutional neural network model with the following configuration:
  - `in_channels=3` (for RGB images)
  - `num_filters=[128, 128, 64, 64, 32]` (number of filters for each convolutional layer)
  - `filter_size=[3, 3, 3, 3, 3]` (filter size for each convolutional layer)
  - `activation='mish'` (activation function used for non-linearity)
  - `stride=1` (stride for the convolutions)
  - `padding='same'` (padding for the convolution layers to maintain input size)
  - `pool_size=(2, 2)` (pooling layer size)
  - `fc_size=256` (fully connected layer size)
  - `num_classes=10` (number of classes for classification)
  - `dropout=0.3` (dropout rate to prevent overfitting)
  - `batch_norm=True` (use batch normalization to improve training)
  - `device=device` (use GPU or CPU as defined earlier)
- Trains the model using the `model_training` function with 30 epochs and a learning rate of 0.001
- Evaluates the trained model on the test set using the `test` function, which computes and prints the test accuracy

**Step 13:**

- Defines the `plot_predictions_grid` function to visualize the model's predictions on a subset of the dataset
- Takes the following parameters:
  - `model`: The trained model to make predictions
  - `dataset`: The dataset containing images and their labels
  - `device`: The device (CPU or GPU) where the model is loaded
  - `num_images`: Number of images to display (default is 30)
- The function:
  - Sets the model to evaluation mode using `model.eval()`
  - Randomly selects `num_images` indices from the dataset
  - Creates a grid of subplots to display the images and their predicted vs. true labels
  - For each image:
    - Loads the image and its true label
    - Runs the image through the model to get the predicted label
    - Converts the image from tensor format to NumPy array for plotting
    - Displays the image along with its predicted and true labels in the grid
  - Tightens the layout and shows the grid of images with predictions
 
**Step 14:**

    - Defines the `plot_first_layer_filters` function to visualize the first layer filters of a Convolutional Neural Network (CNN)
- Takes the following parameter:
  - `model`: The trained model from which to extract and display the first layer filters
- The function:
  - Extracts the first convolutional layer from the model using `list(model.children())[0]`
  - Checks if the first layer has a `weight` attribute (indicating it’s a convolutional layer) and retrieves the filter weights
  - Creates a grid of 8x8 subplots to display the filters
  - Loops over the filters and displays each in grayscale on the subplots
  - Adds a title to the figure: "8x8 Grid of First-Layer Filters"
  - Tightens the layout and shows the grid of filters
 


  # **Result:**

**Model Training and Evaluation Summary:**

After conducting approximately 35 hyperparameter sweeps, we identified a model configuration that achieved the highest training accuracy of 30.55% within 10 epochs.

- num_filters=[128,128,64,64,32]
- kernal_size=[3, 3, 3, 3, 3]
- activation='mish'
- num_dense=256
- dropout=0.3
- batch_norm=True
- learning rate= 0.001

Subsequently, we extended the training of this model to 30 epochs, observing the following performance metrics(Step 12 in the notebook):

- Peak validation accuracy: 36.95% (achieved at epoch 30)

- Final test accuracy: 39.64% (after 30 epochs)

In a final evaluation using the trained model at 30 epochs, we achieved an improved test accuracy of 43.15% on our test dataset, demonstrating the model's enhanced generalization capability with extended training.

**Model Validation on Sampled Test Data:**

To further evaluate the model’s performance, we randomly selected 30 samples from the test dataset and compared the actual class labels with the model’s predictions. The results were as follows:

 - Correct predictions: 15/30 (50% accuracy)

 - Incorrect predictions: 15/30

This manual validation aligns with the model’s overall test accuracy (~43.15%), confirming its classification behavior on unseen data.
