#importing required libraries
import streamlit as st
import pickle
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

loaded_model = pickle. load(open("model.sav", 'rb'))
nlp = spacy.load('en_core_web_lg')

#adding a text area input widget
txt = st.text_area('Enter text: ')

#displaying the text entered by the user
st.write('The score is:', loaded_model.predict(np.array([nlp(txt).vector]))[0])