# Fine-tuning InceptionV4 on Caltech101 Image Classification Dataset

![ezgif com-gif-maker](https://github.com/jmayank23/Caltech101_Deephys_NeuralActivity/assets/27727185/85e410e6-1ac3-4c0f-ac45-57d2a6304b44)

This code fine-tunes an InceptionV4 model on the Caltech101 image classification dataset and exports the neural activity for visualization in Deephys.

## Deephys and its Importance

Deephys is a powerful tool for visualizing the inner workings of neural networks. It provides an intuitive way to explore the learned representations within the model's layers. By leveraging Deephys, we can obtain insights into the decision-making process of the model, which can greatly aid in the understanding, troubleshooting, and optimization of neural networks.

This project particularly shows how to extract neural activity from a trained model and how to prepare it for visualization in Deephys. This involves capturing the model's output and the neural activity from specified layers during the forward pass.

## Code Modularity and Adaptability

The code shared here is designed to be adaptable for different models and datasets. It's broken down into distinct steps, each with its own function, making it easier to understand and modify as per the requirements of different models or datasets.

1. **Data Loading and Preparation**: The process of loading and preparing data is independent of the specific dataset being used, so you can replace `load_data()` with a function to load your own dataset.

2. **Model Training**: The model training function `train_model()` can be modified to fit different models and training configurations.

3. **Extracting Neural Activity for Deephys**: This section uses PyTorch hooks to capture neural activity. The hook is registered to a specific layer in the model, and this can be adapted to any layer in any model.

4. **Deephys Model Definition and Data Export**: The definition of the Deephys model and the export of the data to a Deephys-compatible format are generalized processes. You can easily adjust the `layers` parameter when defining the Deephys model to include the layers you are interested in.

By maintaining modularity and generality in the code, we ensure that it can be adapted to different use cases with minimal changes, making it a robust and flexible starting point for various machine learning projects.


## Detailed Workflow and Code Description

### Fine-tuning InceptionV4 on Caltech101 Image Classification Dataset

This code fine-tunes an InceptionV4 model on the Caltech101 image classification dataset and exports the neural activity for visualization in Deephys.

### Data Loading and Preparation

The Caltech101 dataset is loaded and split into training and testing datasets. The datasets are then loaded into PyTorch data loaders with a batch size of 64:

```python
def load_data():
    ...
    return train_dataset, test_dataset, num_classes
  
train_dataset, test_dataset, num_classes = load_data(test_split=0.2)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
```

### Model Training

The InceptionV4 model is then trained on the training dataset for 11 epochs, with the model checkpoints being saved in a specified path:

```python
num_epochs = 11
model = train_model(train_loader, test_loader, num_classes, num_epochs=num_epochs, checkpoint_path='/content/drive/MyDrive/caltech101/model_checkpoint.pth')
```

### Extracting Neural Activity for Deephys

A hook is registered with the model to extract the neural activity from the penultimate layer. For Inception V4, it was after the forward pass through the global pool layer. Note, to figure out the penultimate layer, it is helpful to `print(model)`\
These activations are stored in the 'model_activity' dictionary, with the key as `linear1`:

```python
model_activity = {}
def get_activation(name):
  def hook(model, input, output):
    model_activity[name] = output.detach()
  return hook

h = model.global_pool.register_forward_hook(get_activation('linear1'))
```

A function is then defined to extract the image data, ground truth labels, and neural activity needed for Deephys:

```python
def extract_activity(testloader, model):
  ...
  return dp_images, dp_gt, dp_activity
```

### Deephys Model Definition and Data Export

The Deephys model is defined with the layers that are to be visualized and then saved:

```python
dp_model = dp.model(
    name = "inception_v4",
    layers = {
        "linear1": model.last_linear.in_features,
        "output": len(classes)
    },
    classification_layer="output"
)

dp_model.save('/content/drive/MyDrive/caltech101/inception_v4.model')
```

Finally, the image data, ground truth labels, and neural activity are converted to a Deephys-compatible format and saved:

```python
dataset_activity = dp.dataset_activity(
    name = "Caltech101",
    category_names = classes,
    images = original_batch,
    groundtruth = IN_gt,
    neural_activity = IN_activity,
    model=dp_model,
    )

dataset_activity.save('/content/drive/MyDrive/caltech101/Caltech101.test')
```

## Screenshots from the Deephys app
1. **Visualization based on category:** Displays the most activated neuron for that class along with images it had the highest activations for, accuracy, false [positive, negative] along with the relevant image (need to scroll down in the app.)
<img width="1512" alt="247984813-1fae4604-08cf-49e6-8e34-66f588bcff5e" src="https://github.com/jmayank23/Caltech101_Deephys_NeuralActivity/assets/27727185/5bec5ca4-0b1e-498f-b609-d6ea7a8a6932">

<br/>

2. **Visualization for each image:** Shows the most activated neurons for that image along with other images that the neuron was also activated for. Also shows ground truth and predictions for that image by the model.
<img width="1507" alt="Screen Shot 2023-06-21 at 9 53 05 PM" src="https://github.com/jmayank23/Caltech101_Deephys_NeuralActivity/assets/27727185/34f8d436-b9de-4a31-aeed-2ee7b59ee83f">

<br/>

3. **Visualization for each neuron:** Presents the images a given neuron was highly activated for (arranged in decreasing order of activation value.)
<img width="1507" alt="Screen Shot 2023-06-21 at 9 52 36 PM" src="https://github.com/jmayank23/Caltech101_Deephys_NeuralActivity/assets/27727185/531eae08-3e6f-48b0-987f-44c8aa59087e">

## Final Notes

This project allows you to train a model on an image classification task and then visualize the neural activity of the model using Deephys. The specific layers of the model that are visualized can be easily adjusted by changing the `layers` parameter when defining the Deephys model. This project can serve as a template to adapt to your own datasets and models, promoting understanding and interpretability in machine learning.

## Acknowledgements
I'd like to extend my gratitude to the creators of [Deephys](https://deephys.org/) for their invaluable contribution to machine learning interpretability. Their tool has significantly facilitated my understanding of neural networks. For more details on Deephys, please visit their [GitHub repository](https://github.com/mjgroth/deephys-aio).
