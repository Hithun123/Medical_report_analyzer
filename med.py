import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import re
import openai
from gtts import gTTS
import soundfile as sf


first_image=Image.open(r'D:\Data Science\DL Deep Learning\medical\pxfuel (1).jpg')
second_image1=Image.open(r'D:\Data Science\DL Deep Learning\medical\aa9eda1b-09c1-4297-a18a-8a6694962e32.png')
second_image2=Image.open(r'D:\Data Science\DL Deep Learning\medical\F2.large.jpg')
second_image3=Image.open(r'D:\Data Science\DL Deep Learning\medical\1-16c0c9c335.jpg')

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
openai.api_key='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'

disease_keywords=['diarrhea','fever', "Cancer","Diabetes","Hypertension","Influenza","Asthma","Heart Disease","Stroke","Alzheimer's Disease","Arthritis",
"Obesity","Depression","Anxiety","COPD","Pneumonia","HIV","AIDS","Malaria","Tuberculosis","Hepatitis","Parkinson's Disease","Multiple Sclerosis","Osteoporosis",
"Autism","Schizophrenia","Eating Disorders","Fibromyalgia","Kidney Disease","Liver Disease","Thyroid Disorders","Endometriosis","Lyme Disease","Melanoma","Leukemia",
"Rheumatoid Arthritis","Ulcerative Colitis","Crohn's Disease","Breast Cancer","Lung Cancer","Prostate Cancer","Colon Cancer","Ovarian Cancer","Pancreatic Cancer",
"Kidney Cancer","Liver Cancer","Brain Tumor","Hodgkin's Lymphoma","Non-Hodgkin Lymphoma","Osteosarcoma","Endometrial Cancer","Cervical Cancer","Stomach Cancer","Esophageal Cancer",
"Bladder Cancer","Testicular Cancer","Thyroid Cancer","Uterine Cancer","Oral Cancer","Skin Cancer","Lymphoma","Myeloma","Fibroids","Gallstones","Diverticulitis",
"Gout","Hemorrhoids","IBS","Menstrual Disorders","PCOS","Sinusitis","Tinnitus","Varicose Veins","Sleep Apnea","Eczema","Psoriasis","Rosacea","Acne","Conjunctivitis",
"Glaucoma","Cataracts","Astigmatism","Nearsightedness","Farsightedness","Macular Degeneration","Retinal Detachment","Diarrhea","Constipation",
"GERD","Ulcers","Pancreatitis","Appendicitis","Colitis","Gastritis","Diverticulosis","Diverticulitis","Gastroenteritis","Lactose Intolerance","Celiac Disease",
"Food Poisoning","IBD","Gastric Cancer","Gallbladder Cancer","Anal Cancer","Intestinal Cancer","Bipolar Disorder","Borderline Personality Disorder",
"PTSD","ADHD","OCD","Anorexia Nervosa","Bulimia Nervosa","Binge Eating Disorder","Insomnia","Narcolepsy","Restless Legs Syndrome","Sleepwalking","Night Terrors",
"Night Sweats","Sleep Paralysis","Nightmares","Teeth Grinding","TMJ","Ankylosing Spondylitis","Lupus","Scleroderma","Paget's Disease of Bone",
"Amyotrophic Lateral Sclerosis","Epilepsy","Migraine","Cluster Headache","Tension Headache","Hormone Imbalance","Menopause","Andropause","Pelvic Inflammatory Disease",
"Vaginal Infections","Vaginal Cancer","Vulvar Cancer"]

def main():
    
    
    st.sidebar.header("Home")
    selected_section = st.sidebar.radio("",["About", "Models", "Upload"])

    if selected_section == "About":
        home_section()

    elif selected_section == "Models":
        examples_section()
    elif selected_section == "Upload":
        upload_section()

def home_section():
    st.header("Medical Report Analyzer")
    st.write("This website application provides users with knowledge about all the disease-related measures they need to take.\nTo receive the appropriate safety measures for your disease, you needed to upload the medical report.")
    st.image(first_image)

def examples_section():
    st.header("Models")
    st.write("Models of Medical Reports are given below....")
    st.image(second_image1,width=300)
    st.image(second_image2,width=300)
    st.image(second_image3,width=300)

    
   

def upload_section():
    st.header("Upload")
    st.warning('Upload the medical report below....')
    uploaded_image=st.file_uploader("Upload the medical report",type=["jpg","png"])
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        gray_img=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
        thresh,bin_img=cv2.threshold(gray_img,180,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        text=pytesseract.image_to_string(bin_img)
        pattern = re.compile(r'\b(?:' + '|'.join(disease_keywords) + r')\b', re.IGNORECASE)
        disease=pattern.findall(text)
        disease=list(set(disease))
        if disease:
            for i in disease:
                prompt=f"what are the precautions needed to take for the disease: '{i}'"
                response=openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.1,
                )
                precaution=response.choices[0].text.strip()
                instructions=gTTS(text=precaution, lang="en", slow=False)
                instructions.save("output.mp3")
                audio_file = "output.mp3"
                data, sample_rate = sf.read(audio_file)
                st.image(uploaded_image)
                result=''
                if st.button('PRECAUTIONS'):
                    result=precaution
                
                
                st.success(result)
                st.subheader("  Audio File:  ")
                st.audio(data, format="audio/mp3", sample_rate=sample_rate) 
                

        else:
            return ('No disease detected')

    



if __name__ == "__main__":
    main()
