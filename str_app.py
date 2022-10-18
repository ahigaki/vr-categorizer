import numpy as np
from pathlib import Path
src_path = Path(__file__).resolve().parent #get the abs path of this script
import joblib

import streamlit as st

def main():
    st.title("Categorizer of Virtual Reality-related Research in Cardiology")
    st.subheader("A machine learning-based model to automatically classify research abstracts into two categories (Type A/B) according to the user of VR-devices.")
    input_txt = st.text_area("Input the ABSTRACT of the research that you want to categorize:", " ")
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
    if X_lr == 1:
        st.text("This Research is 'Type A': Health Care Providers use the VR-device")
    else:
        st.text("This Research is 'Type B': Patinets use the VR-device")

    st.markdown("Copyright: [AkinoriHigaki](https://researchmap.jp/ahigaki?lang=en)")
    

if __name__ == "__main__":
    main()


