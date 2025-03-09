# Image Classifier Project

This project trains a deep learning model using TensorFlow and Keras to classify flowers from images.

##  Project Files
- `Project_Image_Classifier_Project.ipynb` - Jupyter notebook for training and testing the model.
- `predict.py` - Command-line script for predicting flower names from images.
- `flower_classifier.keras` - Saved trained model.
- `label_map.json` - Maps class labels to flower names.
- `assets/` & `test_images/` - Example images for testing.

## How to Use
### **Train the Model (Notebook)**
Run the Jupyter notebook to train the model and save it as `flower_classifier.keras`.

### **Predict Flowers from the Command Line**
```bash
python predict.py test_images/orange_dahlia.jpg --model flower_classifier.keras --category_names label_map.json --top_k 5
