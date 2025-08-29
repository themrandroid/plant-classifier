# Little Botanist

Little Botanist is an interactive **Plant Classification & Exploration App** built with **Streamlit**.  
It uses a deep learning model trained on ~10k plant images across **100 classes**, covering flowers, fruits, trees, shrubs, weeds, and medicinal plants.  

The app allows you to:
- Search for plants by **scientific name** or **common name**  
- Upload an image to get **top-3 AI predictions**  
- Browse random featured plants by category  
- Switch between light/dark themes  


## Features
- **Image Classification** → Upload a plant photo and get AI-powered predictions  
- **Search by Name** → Find plants using either scientific or common names  
- **Plant Explorer** → Browse plants grouped into categories:  
  - Flowers 
  - Fruits & Fruit Plants
  - Common Trees & Shrubs
  - Weeds & Medicinal Plants
- **Dynamic Gallery** → Random selection of plants shown on each app load  
- **Rich Plant Info** → Each plant comes with description + fun fact  


## Model
- Trained on **TensorFlow / Keras** with transfer learning  
- Image size: **224 × 224**  
- Techniques applied:  
  - Data augmentation (flip, rotation, zoom)  
  - Mixup regularization  
- Best run achieved **~60% accuracy** (Val/Test) on 100 classes  
