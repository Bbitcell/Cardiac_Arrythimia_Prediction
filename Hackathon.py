import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit
import os
import subprocess
import sys
import webbrowser
class main:
    def __init__(self):
        st.title("ARRHYTHMIA DETECTION TOOL")
        st.markdown('Only ueed for Ischemic changes, Sinus tachycardy, and Sinus bradycardy detection.')
        st.header("Train Model")

        upload_csv = st.file_uploader("Upload CSV File")
        if upload_csv is not None:
            df_arrhythmia = pd.read_csv(upload_csv)
            df_arrhythmia = df_arrhythmia[ df_arrhythmia["class"].isin([1, 2 ,5, 6])]
            st.write(df_arrhythmia)
            df_arrhythmia = df_arrhythmia.fillna(0)
            df_arrhythmia_4_1 = df_arrhythmia[['heartrate','chDI_TwaveAmp','chV6_TwaveAmp','chAVR_QRSTA','chV4_TwaveAmp','chV5_QRSTA','class']]

            # Split data to X and Y
            X = df_arrhythmia_4_1.drop(['class'], axis=1)

            y = df_arrhythmia_4_1['class']

            # Model Training 
            arrhythmia_RandomForestClass2 = RandomForestClassifier(random_state=0,n_estimators=20)
            arrhythmia_RandomForestClass2.fit(X, y)

            st.write("Complete")



            st.header("Patient Information")
            int_Heartrate = st.number_input("Heartrate")
            int_chDI_TwaveAmp = st.number_input("chDI_TwaveAmp")
            int_chV6_TwaveAmp = st.number_input("chV6_TwaveAmp")
            int_chAVR_QRSTA = st.number_input("chAVR_QRSTA Rate")
            int_chV4_TwaveAmp = st.number_input("chV4_TwaveAmp")
            int_chV5_QRSTA = st.number_input("chV5_QRSTA")

            if st.button("Enter"):

                use = {'heartrate': [int_Heartrate],'chDI_TwaveAmp' : [int_chDI_TwaveAmp],'chV6_TwaveAmp' : [int_chV6_TwaveAmp],'chAVR_QRSTA' : [int_chAVR_QRSTA],'chV4_TwaveAmp' : [int_chV4_TwaveAmp],'chV5_QRSTA' : [int_chV5_QRSTA]}
                arrhythmia_RandomForestClass2 = RandomForestClassifier(random_state=0,n_estimators=20)
                arrhythmia_RandomForestClass2.fit(X, y)
                user_data = pd.DataFrame(data = use) 
                predications = arrhythmia_RandomForestClass2.predict(user_data)


                if predications[0] == 1:
                    pred = "Normal"
                elif predications[0] == 2:
                    pred = "Ischemic changes"
                elif predications[0] == 5:
                    pred = "Sinus tachycardy"
                elif predications[0] == 6:
                    pred = "Sinus bradycardy"
                st.header("Detected: " + pred)

                   

if __name__ == "__main__":
    main()