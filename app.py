
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
import pickle
import pandas as pd
import transformers as ppb
import tensorflow as tf
import torch
import numpy as np

st.title("Social Media Vigilancer")
st.write("Identify Hatespeech")

max_len=98
# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Classify the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

user_input = st.text_input("Enter text in meme")

# Display the entered text
if user_input is not None:
    st.write("You entered:", user_input)

    # Perform inference
    # TODO: Preprocess the image if necessary

    # TODO: Pass the preprocessed image through the model
    # e.g., output = model(preprocessed_image)

    # TODO: Perform post-processing and get the predicted class label
    # e.g., predicted_label = process_output(output)

if uploaded_file is not None and user_input is not None:
    inception = InceptionV3(weights='imagenet')
    image = image.resize((299, 299))
    x = img_to_array(image) #convert grey scale of RGB array of (0,255)
    x = np.array([x]) # converting into list of each x
    x = preprocess_input(x) #normalize and remove RGB mean range btw[-1,1]
    preds = inception.predict(x)# predcting probabilty of each calss for the x wat in (299,299,3) height,width,dim(RGB)
    predicted = decode_predictions(preds, top = 5)[0]
    data=user_input
    for j in range(0, 5):
        data=data+" "+predicted[j][1]
    dict={'text':[data]}
    df = pd.DataFrame(dict)# creating a dataframe
    ## for BERT 
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights) #tokenizer form hugging face
    bert_model = model_class.from_pretrained(pretrained_weights)# take  all previsous layers hidden or o/p
    tokenize = tokenizer_class.from_pretrained(pretrained_weights)
    tokenized = df['text'].apply((lambda x: tokenize.encode(str(x), add_special_tokens=True)))# work with daatframe
    
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    
    
    # Extract Attention Masks(tell which token to ignore and to process in form of 0 and 1 for bert model)
    attention_mask = np.where(padded != 0, 1, 0)

    # Here we just convert the data to tensors
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    # no gradient while forwarding propagation
    with torch.no_grad():
        last_hidden_states = bert_model(input_ids, attention_mask=attention_mask)

    # Extracting features of the CLS tokens
    feat = last_hidden_states[0][:,0,:].numpy()
    
    #ML models
    with open('lr_clf1.pkl', 'rb') as f:
        lr_clf = pickle.load(f)
    with open('svm.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('rfm.pkl', 'rb') as f:
        rfm = pickle.load(f)
    
    lstm = tf.keras.models.load_model('lstm.h5')
    bilstm = tf.keras.models.load_model('bilstm.h5')

    feat_reshaped = np.reshape(feat, (-1, 1, 768)) # for 3d shape

if st.button("Predict"):
        pred1 = lr_clf.predict(feat)
        pred2=svm.predict(feat)
        pred3=rfm.predict(feat)
        pred4=lstm.predict(feat_reshaped)
        pred5=bilstm.predict(feat_reshaped)
        st.write("Prediction of Logistic Regression:", pred1[0])
        st.write("Prediction of SVM:", pred2[0])
        st.write("Prediction of Random Forest:", pred3[0])
        st.write("Prediction of LSTM:", pred4[0])
        st.write("Prediction of BILSTM:", pred5[0])
