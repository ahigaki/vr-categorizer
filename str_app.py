import numpy as np
from pathlib import Path
src_path = Path(__file__).resolve().parent #get the abs path of this script
import joblib

from PIL import Image
img_A = Image.open(str(src_path) + '/img_A.png')
img_B = Image.open(str(src_path) + '/img_B.png')

import streamlit as st

def main():
    st.title("Categorizer of Virtual Reality-related Research in Cardiology")
    st.subheader("A machine learning-based model to automatically classify research abstracts into two categories (Type A/B) according to the user of VR-devices.")
    input_txt = st.text_area("Input the ABSTRACT of the research that you want to categorize:",
                             " ")
    #input_txt = input("input abstract:")
    input_txt_lst = [input_txt]

    ##########TOKENIZATION##########
    tfid_path = str(src_path) + '/tfid_model.sav'
    vect = joblib.load(tfid_path)

    vect_txt = vect.transform(input_txt_lst)
    #feat_names = vect.get_feature_names()

    ##########BINARY-PREDICTION##########
    lr_path = str(src_path) + '/lr_model.sav'
    lr = joblib.load(lr_path)

    X_lr = lr.predict(vect_txt)
    p_lr = lr.predict_proba(vect_txt).round(3)

    st.subheader("Results...")

    if X_lr == 1:
        st.text("This Research is 'Type A': Health Care Providers use the VR-device")
        st.image(img_A)
        st.text("Probability of Type A:" + str(p_lr[:,1]))
    else:
        st.text("This Research is 'Type B': Patinets use the VR-device")
        st.image(img_B)
        st.text("Probability of Type B:" + str(p_lr[:,0]))

    keys = ["virtual", "reality", "vr"]
    if input_txt == " ":
        pass
    elif set(keys).isdisjoint(set(input_txt.split())):
        st.write('<span style="color:red; font-size:150%">but, maybe this research is not related to VR</span>', unsafe_allow_html=True)
    

    st.markdown("Copyright: [AkinoriHigaki](https://researchmap.jp/ahigaki?lang=en)")
    

if __name__ == "__main__":
    main()


