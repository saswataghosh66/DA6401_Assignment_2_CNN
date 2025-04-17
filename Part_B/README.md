# **Aim:**

To improve classification accuracy on the [iNaturalist](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) dataset using transfer learning by fine-tuning a pre-trained deep learning model (e.g., ResNet50, VGG) from ImageNet instead of training from scratch.

# **Key Steps to Achieve the Aim:**

### Step 1:

In this section, essential libraries are imported for deep learning, data handling, preprocessing, model training, evaluation, and experiment tracking. 

- **Deep Learning**: Libraries like `torch`, `torch.nn`, and `torch.optim` are utilized for building, training, and optimizing neural networks.
- **Data Handling**: `ImageFolder`, `transforms`, and `DataLoader` are used for loading and preprocessing the dataset.
- **Experiment Tracking**: `wandb` is included to track experiments and visualize metrics.
- **Visualization**: `matplotlib` and `numpy` help in visualizing data and results during the training process.
- **Utilities**: `SimpleNamespace` is used to manage configurations, and `random` is used for generating random samples or operations. 

These libraries support the entire workflow, from model creation to evaluation and tracking.

### Step 2:

 Function: `pretrain_resnet50`

This function customizes the freezing of layers in a pre-trained ResNet50 model, allowing for different strategies to control which layers are fine-tuned during training.

 Parameters:
- **`strategy`** (str): Specifies how layers are frozen:
  - `'freeze_last'`: Freezes all layers except the fully connected (fc) layer.
  - `'freeze_k_percent'`: Freezes a percentage of layers based on `freeze_percent`.
  - `'freeze_except_bn_and_fc'`: Freezes all layers except batch normalization and fc layers.
  
- **`freeze_percent`** (float, optional): The percentage of layers to freeze (only for `'freeze_k_percent'` strategy).

How It Works:
1. Loads the pre-trained ResNet50 model.
2. Modifies the output layer to match the number of classes (10).
3. Applies the specified freezing strategy:
   - **`'freeze_last'`**: Freezes all layers except the fc layer.
   - **`'freeze_k_percent'`**: Freezes a percentage of layers.
   - **`'freeze_except_bn_and_fc'`**: Freezes all layers except batch normalization and fc.
4. Moves the model to the specified device (CPU or GPU).
5. Returns the modified model.

### Step 3:
Visualizing Random Images from a Class

This code randomly selects a class from the training dataset directory and displays 10 random images from that class. The purpose is to visually inspect the images and verify the diversity and quality of the images in a particular class.

-  A random class is selected from the available class folders.
- 10 random images are chosen from that class folder.
- The images are displayed in a row, with the class name as the title above the images.

This can be useful for quickly examining the contents of your dataset and ensuring the images are correctly labeled.

### Step 4:

Data Loading with Optional Augmentation

The `data_load` function loads the training dataset, applies optional augmentation, and splits the data into training and validation sets.

Parameters:
- **`train_data_dir`** (str): Path to the training images directory.
- **`data_augmentation`** (str): Apply augmentation if `'Yes'` (random cropping, flip, jitter, rotation), otherwise `'No'`.

Steps:
1. **Base Transformations**: Resizes, converts to tensor, and normalizes images.
2. **Data Augmentation (Optional)**: Applies augmentation techniques if `'Yes'`.
3. **Dataset Creation**: Uses `ImageFolder` to load images organized by class.
4. **Data Splitting**: Splits data into 80% training and 20% validation sets.
5. **Data Loaders**: Returns `train_loader` and `val_loader` for batching.
6. **Returns**: DataLoaders, dataset, and split indices for training and validation.

### Step 5:

 Model Training with WandB Logging

The `model_training` function trains a model on the training dataset, evaluates on the validation dataset, and logs the performance to Weights & Biases (WandB), if enabled. It also plots the accuracy over epochs after training.

Parameters:
- **`model`**: The model to train.
- **`train_data`**: The training data loader.
- **`val_data`**: The validation data loader.
- **`epochs`** (int): Number of training epochs (default: 10).
- **`device`** (str): Device for training (`'cuda'` or `'cpu'`).
- **`lr`** (float): Learning rate (default: 0.001).
- **`use_wandb`** (bool): Enable WandB logging (default: `True`).

Steps:
1. **Initialize Model**: Moves the model to the specified device.
2. **Loss & Optimizer**: Sets up `CrossEntropyLoss` and Adam optimizer.
3. **WandB Setup**: Logs training details if `use_wandb` is `True`.
4. **Training Loop**: Computes loss and accuracy for each batch and updates model weights.
5. **Validation Loop**: Calculates validation accuracy without weight updates.
6. **WandB Logging**: Logs training and validation metrics after each epoch.
7. **Training Summary**: Plots the training and validation accuracy.
8. **Output**: Returns the trained model.

### Step 6:

 Transfer Learning Strategies

We implemented and compared three fine-tuning approaches:

 1. Frozen Feature Extractor
- **Method:** Freeze all backbone layers, train only classifier
- **Layers frozen:** All except final fully-connected layer
- **Purpose:** Baseline transfer learning approach

 2. Partial Freezing (25%)
- **Method:** Freeze first quarter of network
- **Layers frozen:** ~25% of early convolutional blocks
- **Purpose:** Balance feature preservation and adaptation

 3. BatchNorm + Classifier Tuning  
- **Method:** Unfreeze BatchNorm layers + classifier
- **Trainable parameters:**
  - All BatchNorm layers
  - Final classification head
- **Purpose:** Adapt feature normalization to target domain

### Step 7:

 I selected the freeze_except_bn_and_fc strategy—where only the batch normalization layers and the final classifier layer are kept trainable—as it yielded the 
 best performance during initial testing.
 I trained the ResNet50 model using this strategy for 30 epochs

### Step 8:

- Loads validation images from the specified directory (`test_data_dir`)
- Randomly selects a class from the validation data folders
- Samples 10 images from the randomly selected class
- Displays the 10 selected images in a 1x10 grid without axis labels
- Titles each image with the class name

### Step 9:

- Defines the function `test_data_load` to load and preprocess the test dataset
- Applies a series of transformations (resizing, normalization, and optional data augmentation)
- Augmentation includes random resized crop, horizontal flip, color jitter, and random rotation
- Returns a DataLoader for the test dataset to batch the images for evaluation
- Calls the function with `data_augmentation` set to "No"
- Prints the total number of images in the test dataset and the number of batches in the test DataLoader

### Step 10:

- Defines a function `test` to evaluate the model's performance on the test dataset
- Switches the model to evaluation mode using `model.eval()`
- Initializes counters for correct predictions and total samples
- Loops over the test data and performs predictions without calculating gradients (`torch.no_grad()`)
- Calculates the number of correct predictions and the total number of samples
- Computes the test accuracy as the percentage of correct predictions
- Prints and returns the test accuracy

### Step 11:

- Defines the `plot_predictions_grid` function to visualize the model's predictions on a subset of the dataset
- Takes the following parameters:
  - `model`: The trained model to make predictions
  - `dataset`: The dataset containing images and their labels
  - `device`: The device (CPU or GPU) where the model is loade
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
 

    
# **Results:**

ResNet50 Fine-Tuning Strategies and Results
In step 6, I fine-tuned a pre-trained ResNet50 model using three different layer freezing strategies, each trained for 5 epochs. The strategies and their corresponding training accuracies are as follows:

- freeze_last: All layers of the model were frozen except the final fully connected (fc) layer.
   - Training Accuracy: 75.58%
   - Validation Accuracy: 75.50%
- freeze_k_percent: The first 25% of the model's parameters were frozen, while the remaining 75% were trainable.
   - Training Accuracy: 64.02%
   - Validation Accuracy: 59.20%
- freeze_except_bn_and_fc: All layers were frozen except for the batch normalization layers and the final classifier layer.
   - Training Accuracy: 82.92%
   - Validation Accuracy: 80.30%
 
These results highlight that allowing the batch normalization layers to remain trainable (freeze_except_bn_and_fc) resulted in the highest training accuracy, indicating better adaptation to the new dataset.

### Extended Training with Optimal Strategy:

In Step 7, based on the results from the previous experiments, I selected the freeze_except_bn_and_fc strategy—where only the batch normalization layers and the final classifier layer are kept trainable—as it yielded the best performance during initial testing.

I trained the ResNet50 model using this strategy for 30 epochs, which resulted in:

  - Training Accuracy: 88%

  - Validation Accuracy: 89%

This demonstrates that with extended training, the model was able to generalize well to unseen data, as evidenced by the high validation accuracy closely matching the training accuracy.


### Model Evaluation on Test Set:

After training the ResNet50 model for 30 epochs using the freeze_except_bn_and_fc strategy, I evaluated its performance on the unseen test dataset. The model achieved a test accuracy of 86.25%, indicating strong generalization ability and consistent performance across training, validation, and test sets.

This result confirms that selectively fine-tuning batch normalization and classifier layers is an effective strategy for transfer learning with ResNet50 on this dataset.

### Model Validation on Sampled Test Data:

To further assess the model’s performance, a random subset of 30 samples was selected from the test dataset. The model's predictions were compared against the actual class labels.

Correct Predictions: 26 out of 30

Accuracy on Sampled Data: 86.67%

This manual check is consistent with the overall test accuracy of 86.25%, reinforcing the model’s reliable performance on unseen data and its ability to generalize well beyond the training and validation sets.
