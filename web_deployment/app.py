import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from segmentation_models.losses import JaccardLoss
from segmentation_models.metrics import IOUScore
import os
from io import BytesIO
import base64

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    loss_func = JaccardLoss(per_image=True)
    metric = IOUScore(per_image=True)

    model = tf.keras.models.load_model('/app/web_app_background_removal/Unet_model', compile=False)
    model.compile(loss = loss_func, optimizer="adam", metrics=[metric])
    return model

with st.spinner('Loading the tool..'):
    model=load_model()

st.write("""
        # Background Remover Tool   
         This is a tool that can be used to remove background from your images. It is based on semantic segmentation and  works best for **portrait images that contains a single human in focus.**

         *This was created as a personal project so save some criticsm please :)*

         *The model actually works on 128X128 size, so if you have quite a large image then there will be heavy loss in downsampling/upsampling and the output won't be very good.
         Also, the model was trained on a very selective dataset of front-facing, portrait images of humans due to computational limitations, therefore might give sub-optimal performance in other cases.*

         Created by - ***[UMAR MASUD](https://umar07.github.io)***
	 
         [Source Code](https://github.com/umar07/Background_Removal_Semantic_Segmentation)
	  	 
         """
         )

file = st.file_uploader("Please upload you image below", type=["jpg", "png", "jpeg"])


def predict(img_test_orig, model):
    
    size = (128,128)    
    h, w = img_test_orig.shape[0:2]

    img_test_resized=cv2.resize(img_test_orig, size)
    img_test=np.asarray(img_test_resized)/255.0
    img_test = img_test[np.newaxis,...]

    pred_img=model.predict(img_test)

    pred_img = np.squeeze(pred_img)

    result = img_test_resized.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    pred_img_copy = pred_img.copy()
    pred_img_copy[pred_img_copy<0.5] = 0
    pred_img_copy[pred_img_copy>=0.5] = 255
    print("Shape of result: " , result.shape)
    # print("Shape of pred_img_copy: " , pred_img_copy.shape)

    result[:, :, 3] = pred_img_copy
    result = cv2.resize(result, (h,w))
    # print("bg_removed shape: " , result.shape)
    
    return result

# https://discuss.streamlit.io/t/how-to-download-image/3358/4
def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download="bg-removed.png">DOWNLOAD RESULT</a>'
	return href


if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)

    st.image(image, use_column_width='auto')
    with st.spinner('Removing background..'):
        img = np.asarray(image)
        pred_img = predict(img, model)
        st.success("Background Removed")
        st.image(pred_img, use_column_width='auto')
    
        # st.download_button(
        # label="Download Image",
        # data=,
        # file_name='{}-bg-removed.jpg'.format(file.name.split('.')[0]))

    pred_result = Image.fromarray(pred_img)
    st.write(get_image_download_link(pred_result), unsafe_allow_html=True)
