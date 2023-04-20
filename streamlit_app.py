import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
import csv
import torch
from model import build_model
from class_names import class_names
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image


link_checkpoint = 'resnet152_best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=len(class_names))
model.to(device)
model.load_state_dict(torch.load(link_checkpoint,map_location=lambda storage, loc: storage)['model_state_dict'])
model.eval()

size_img = (224,224)
transform = A.Compose(
        [
            A.Resize(size_img[0], size_img[1], interpolation=1, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2()
        ])

st.set_page_config(
    page_title="Testing",
    layout="wide"
)
_,col1, _ = st.columns([1,4, 1])

with col1:
    st.title("Test API! 👋")

    st.markdown("<h1 style='text-align: center; '>Predict car company</h1>", unsafe_allow_html=True)
    use_form =  True
    if use_form:
        with st.form("my-form", clear_on_submit=True):
            my_images = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button("Upload")
            if submitted:
                st.success("Successfully upload file")
    else:
        my_images = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    st.info(f"Uploaded {len(my_images)} files")

    for my_image in my_images:
        if my_image is not None:
            
            image = Image.open(my_image)

            image = np.array(image)

            image = transform(image=image)["image"]
        
            data  = image.to(device)[None, :]
            output = model(data)
 
            output_softmax = torch.softmax(output,dim=1)
            sort_output = torch.sort(output_softmax,descending=True)
            
            check = sort_output[1][0]
            probability = sort_output[0][0]
            
            st.write("{}: {:.2f}%".format(class_names[check[0]],probability[0]*100))
            st.image(my_image)
            # print("{}: {:.4f} | {}: {:.4f} | {}: {:.4f}".format(
            #         class_names[check[0]],probability[0],
            #         class_names[check[1]],probability[1],
            #         class_names[check[2]],probability[2]))
            
            
                        
            