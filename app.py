import streamlit as st
from model import Model

model_class = Model()
model = model_class.load_model('C:\\Users\\Erickson Neil Ruaro\\Documents\\GitHub\\Dog-Breed-Classifier\\model\\20210704-01-M1625361368-full-image-set-mobilenetv2-Adam.h5')
print('model loaded')
# file = st.file_uploader("Select an image to classify")
# data = Model().create_data_batches(test_data=True, X=file)
