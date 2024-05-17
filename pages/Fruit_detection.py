# from navigation import make_sidebar
# make_sidebar()
# import pandas as pd
# import os
# from os import listdir
# import streamlit as st
# import tensorflow as tf
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)
# from PIL import Image
# from utils import stream_data, spinner_call, remark, get_cpu_memory_usage
# from math import ceil
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
# IMG_SIZE  = (240, 240)
# IMG_SHAPE = IMG_SIZE +(3,)

# import shap


# # st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
# st.sidebar.title('Rotten and Fresh Fruit')
# #1. load model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('rottenvsfresh.h5')

# loaded_model = load_model()

# def draw_column_chart(data):
#     # Convert data into a DataFrame
#     df = pd.DataFrame(data, columns=['Rating', 'Fruit'])

#     # Plotting the data
#     plt.figure(figsize=(14, 8))
#     barplot = sns.barplot(hue='Fruit', y='Rating', data=df, palette=['red' if v == 0 else 'green' for v in df['Rating']])
#     plt.xlabel('Fruits')
#     plt.ylabel('Rating')
#     plt.title('Fruit Ratings')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     # Display the plot using Streamlit
#     st.pyplot(plt)

# class_names = ['Rotten banana', 'Fresh apples',  'Rotten oranges', 'Fresh banana', 'Rotten apples', 'Fresh oranges']

# st.header("Rotten and Fresh Fruit Detection Application")
# st.write_stream(stream_data('This Application utilises machine learning model can identify up to 3 types of fruits apple,\
#                              banana, and orange where fresh or rotten'))

# uploaded_file = st.file_uploader("Upload an image of fruit to check the freshness...", type=['jpg', 'jpeg', 'png'])





# directory = r'assets'
# files = listdir(directory)

# def initialize():    
#     df = pd.DataFrame({'file':files,
#                     'Select':[False]*len(files),
#                     'label':['']*len(files)})
#     df.set_index('file', inplace=True)
#     return df

# if 'df' not in st.session_state:
#     df = initialize()
#     st.session_state.df = df
# else:
#     df = st.session_state.df 



# def update (image, col): 
#     df.at[image,col] = st.session_state[f'{col}_{image}']
#     if st.session_state[f'Select_{image}'] == False:
#                  df.at[image,'label'] = ''

# batch = files[(1-1)*10 : 1*10]

# grid = st.columns(8)
# col = 0
# image_files = os.listdir(directory)
# for image in batch:
#     with grid[col]:
#         st.image(f'{directory}\{image}', caption=os.path.splitext(image)[0])
#         st.checkbox("Select", key=f'Select_{image}', 
#                     value = df.at[image,'Select'], 
#                     on_change=update, args=(image,'Select'))
#     col = (col + 1) % 8

# st.write('## Prediction')

# def explainer_f(loaded_model , img_g):
#         # Load SHAP explainer
#         explainer = shap.DeepExplainer(loaded_model , img_g)

#         # Compute SHAP values
#         shap_values = explainer.shap_values(img_g)

#         # Display SHAP image plot
#         plt.figure(figsize=(6, 6))
#         shap.image_plot(shap_values, img_g, show=False)
#         plt.title('SHAP Explanations')
#         plt.axis('off')
#         st.pyplot(plt)

# def runner(imgs):
#         img = Image.open(imgs).resize(IMG_SIZE)
#         img = np.array(img, dtype=np.float32)
#         img_g = np.expand_dims(img, axis=0)
#         custom_predict = loaded_model.predict(img_g)

#         predicted_class_index = np.argmax(custom_predict)
#             # Get the predicted class label
#         predicted_label = class_names[predicted_class_index]


#         # Display the uploaded image and the predictions
#         if round(np.max(custom_predict[0]*100))>50:
#             st.write(f'<p style="font-size:26px; color:green;"> {class_names[np.argmax(custom_predict[0])]}  ({round(np.max(custom_predict[0]*100))}% confidence)</p> ', unsafe_allow_html=True)
#         else:
#             argsorts = np.argsort(custom_predict[0])
#             sorts = np.sort(custom_predict[0])
#             st.write(f'<p style="font-size:26px; color:red;"> {class_names[argsorts[-1]]}  ({round(sorts[-1]*100)}% confidence) OR {class_names[argsorts[-2]]}  ({round(sorts[-2]*100)}% confidence)</p> ', unsafe_allow_html=True)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.header("Fruit Detected")
#             st.write_stream(stream_data(f'The predected Fruit is {class_names[np.argmax(custom_predict[0])]}'))
#             st.image(img/255,  use_column_width=False)
#             st.write('Recomendation:')
#             if 'Fresh' in class_names[np.argmax(custom_predict[0])]:
#                 st.write(f'<p style="font-size:26px; color:green;">The fruit is fresh and good to consume. You should accept it.</p>', unsafe_allow_html=True)
#             elif 'Rotten' in class_names[np.argmax(custom_predict[0])]:
#                 st.write(f'<p style="font-size:26px; color:red;">The fruit is rotten and unfit for consumption. You should reject it.</p>', unsafe_allow_html=True)


#         with col2:
#             st.header("Probability of Detection")
#             probabilities = custom_predict[0]
#             # Print predictions
#             predictions = []
#             for i in range(len(class_names)):
#                 class_label = class_names[i]
#                 class_probability = probabilities[i]
#                 predictions.append((class_probability, class_label))

#             draw_column_chart(predictions)
#         return class_names[np.argmax(custom_predict[0])], round(np.max(custom_predict[0]*100)), img_g

# if df['Select'] is not None:
#     try:
#         file_dict = dict(df['Select']==True)
#         # Find the key with the value True
#         true_key = next((key for key, value in file_dict.items() if value), None)
#         print(true_key)
#         print(directory+'/'+true_key)

#         labels, percent, img_g = runner(directory+'/'+true_key)
#         explainer_f(loaded_model , img_g)
#     except TypeError as e:
#          print(e)
# # If an image is uploaded, display it and make a prediction
     
# if uploaded_file is not None:

#     labels, percent, img_g = runner(uploaded_file)
#     st.write(f'<p style="font-size:26px; color:black;">The model  has detected that the fruit is likely to be {labels.lower()} with a confidence of {percent}%.</p>', unsafe_allow_html=True)

#     explainer_f(loaded_model , img_g)
        
# Import necessary libraries
import os
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil
import shap

# Import custom utilities
from utils import stream_data, spinner_call, remark, get_cpu_memory_usage
from navigation import make_sidebar

# Configuration for Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Constants
IMG_SIZE = (240, 240)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_NAMES = ['Rotten banana', 'Fresh apples', 'Rotten oranges', 'Fresh banana', 'Rotten apples', 'Fresh oranges']
ASSETS_DIR = 'assets'
BATCH_SIZE = 10

# Sidebar setup
make_sidebar()
st.sidebar.title('Rotten and Fresh Fruit')

# Page setup
st.header("Rotten and Fresh Fruit Detection Application")
st.write_stream(stream_data(
    'This Application utilizes a machine learning model to identify up to 3 types of fruits: apple, banana, and orange, determining if they are fresh or rotten.'
))

# Load model with caching
@st.cache_resource
def load_model():
    """
    Loads the pre-trained Keras model for fruit classification.

    Returns:
        loaded_model (tf.keras.Model): The loaded Keras model.
    """
    return tf.keras.models.load_model('rottenvsfresh.h5')

loaded_model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image of fruit to check the freshness...", type=['jpg', 'jpeg', 'png'])

# Initialize dataframe
def initialize_df(files):
    """
    Initializes a DataFrame to keep track of file selections and labels.

    Args:
        files (list): List of file names in the assets directory.

    Returns:
        df (pd.DataFrame): Initialized DataFrame with file names, selection status, and labels.
    """
    df = pd.DataFrame({'file': files, 'Select': [False] * len(files), 'label': [''] * len(files)})
    df.set_index('file', inplace=True)
    return df

# Initialize session state for DataFrame if not already initialized
if 'df' not in st.session_state:
    files = os.listdir(ASSETS_DIR)
    st.session_state.df = initialize_df(files)

df = st.session_state.df

# Update dataframe based on user interaction
def update_selection(image, col):
    """
    Updates the selection status and label for a given image in the DataFrame.

    Args:
        image (str): Image file name.
        col (str): Column name to update (typically 'Select').
    """
    df.at[image, col] = st.session_state[f'{col}_{image}']
    if not st.session_state[f'Select_{image}']:
        df.at[image, 'label'] = ''

# Display images in a grid
def display_image_grid(files, batch_size):
    """
    Displays images in a grid layout for user selection.

    Args:
        files (list): List of file names to display.
        batch_size (int): Number of images to display in a batch.
    """
    batch = files[:batch_size]
    grid = st.columns(8)
    col = 0
    for image in batch:
        with grid[col]:
            st.image(f'{ASSETS_DIR}/{image}', caption=os.path.splitext(image)[0])
            st.checkbox("Select", key=f'Select_{image}', value=df.at[image, 'Select'], on_change=update_selection, args=(image, 'Select'))
        col = (col + 1) % 8

# Display the initial grid of images
display_image_grid(df.index.tolist(), BATCH_SIZE)

# Function to draw column chart
def draw_column_chart(data):
    """
    Draws a column chart for the given data.

    Args:
        data (list): List of tuples containing probability and fruit label.
    """
    df = pd.DataFrame(data, columns=['Probability', 'Fruit'])
    plt.figure(figsize=(14, 8))
    sns.barplot(hue='Fruit', y='Probability', data=df, palette=['red' if v == 0 else 'green' for v in df['Probability']])
    plt.xlabel('Fruits')
    plt.ylabel('Probability')
    plt.title('Fruit Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

# SHAP explainer function
def explain_with_shap(model, img_g):
    """
    Generates and displays SHAP explanations for the given image.

    Args:
        model (tf.keras.Model): The trained Keras model.
        img_g (np.ndarray): Preprocessed image array.
    """
    explainer = shap.DeepExplainer(model, img_g)
    shap_values = explainer.shap_values(img_g)
    plt.figure(figsize=(6, 6))
    shap.image_plot(shap_values, img_g, show=False)
    plt.title('SHAP Explanations')
    plt.axis('off')
    st.pyplot(plt)

# Prediction runner
def make_prediction(image_path):
    """
    Runs the prediction on the given image and displays the results.

    Args:
        image_path (str): Path to the image file.

    Returns:
        predicted_label (str): Predicted class label.
        confidence (float): Prediction confidence percentage.
        img_g (np.ndarray): Preprocessed image array.
    """
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_g = np.expand_dims(img_array, axis=0)
    predictions = loaded_model.predict(img_g)
    predicted_class_index = np.argmax(predictions)
    predicted_label = CLASS_NAMES[predicted_class_index]
    confidence = round(np.max(predictions[0]) * 100)

    st.write(f'<p style="font-size:26px; color:{"green" if confidence > 50 else "red"};">{predicted_label} ({confidence}% confidence)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Fruit Detected")
        st.write_stream(stream_data(f'The predicted fruit is {predicted_label}'))
        st.image(img_array / 255, use_column_width=False)
        st.write('Recommendation:')
        if 'Fresh' in predicted_label:
            st.write('<p style="font-size:26px; color:green;">The fruit is fresh and good to consume. You should accept it.</p>', unsafe_allow_html=True)
        else:
            st.write('<p style="font-size:26px; color:red;">The fruit is rotten and unfit for consumption. You should reject it.</p>', unsafe_allow_html=True)

    with col2:
        st.header("Probability of Detection")
        probabilities = [(prob, CLASS_NAMES[i]) for i, prob in enumerate(predictions[0])]
        draw_column_chart(probabilities)

    return predicted_label, confidence, img_g

# Execute prediction and explanation if file is selected
selected_files = df[df['Select']].index.tolist()
if selected_files:
    selected_file = selected_files[0]
    labels, percent, img_g = make_prediction(f'{ASSETS_DIR}/{selected_file}')
    explain_with_shap(loaded_model, img_g)

# Execute prediction and explanation if file is uploaded
if uploaded_file is not None:
    labels, percent, img_g = make_prediction(uploaded_file)
    st.write(f'<p style="font-size:26px; color:black;">The model has detected that the fruit is likely to be {labels.lower()} with a confidence of {percent}%.</p>', unsafe_allow_html=True)
    explain_with_shap(loaded_model, img_g)
