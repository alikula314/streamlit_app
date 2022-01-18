import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

# Sayfa Bilgileri
st.set_page_config(page_title="Kredi Uygunluğu", page_icon=":bank:", layout="wide")

st.markdown("<h1 style='text-align: center; font-size: 40px;'>Arat Banka Hoşgeldiniz!</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 20px;'>Aşağıda verilen gerekli bilgileri girerek müşterinin kredi uygunluğuna karar verebilirsiniz.</h1>", unsafe_allow_html=True)
st.markdown("---")

# Eğitilmiş modeli yüklemek
pickle_in = open('classifier.pkl', 'rb') 
model = pickle.load(pickle_in)


value_eur_input = st.slider(label = 'Player Market Value(euro)', min_value = 0, max_value = 400000000)
age_input = st.slider(label = 'Player Age' , min_value = 15, max_value = 45)
height_cm_input = st.slider(label = 'Player Height(cm)', min_value = 140, max_value = 220)
weight_kg_input = st.slider(label = 'Player Weight(kg)', min_value = 50, max_value = 130)
league_level_input = st.slider(label = 'Your Club League Level', min_value = 1, max_value = 5)
international_reputation_input = st.slider(label = 'Player International Reputation', min_value = 1, max_value = 5)
release_clause_eur_input = st.slider(label = 'Player Release Clause(euro)', min_value = 0, max_value = 500000000)
contract_left_days_input = st.slider(label = 'Player Contract Left(day)', min_value = 0, max_value = 2000)
days_at_club_input = st.slider(label = 'Player Elapsed Time In Club(day)', min_value = 0, max_value = 2500)
club_name_input = st.selectbox(label = 'Your Club Name', options = ('FC Barcelona', 'Real Madrid CF'))
league_name_input = st.selectbox(label = 'Your Clubs League Name', options = ('Spain Primera Division', 'German 1. Bundesliga'))
club_position_input = st.selectbox(label = 'Player Position', options = ('CF','LW','SUB','ST','GK'))
nationality_name_input = st.selectbox(label = 'Player Nationality Name', options = ('Argentina', 'Portugal'))
preferred_foot_input = st.selectbox(label = 'Player Preferred Foot', options = ("Left", "Right"))
work_rate_input = st.selectbox(label = 'Player Work Rate ', options = ("Low", "Medium", "High"))
body_type_input = st.selectbox(label = 'Player Body Type', options = ("Normal", "Lean", "Stocky" , "Unique"))


st.markdown("<h1 style='text-align: center; font-size: 40px;'>Summary:</h1>", unsafe_allow_html=True)
summary_dictionary = {'Player Market Value': value_eur_input,  'Player Age': age_input, 'Player Height': height_cm_input, 'Player Weight': weight_kg_input,  'Your Club League Level': league_level_input,'Player International Reputation': international_reputation_input, 'Player Release Clause': release_clause_eur_input, 'Player Contract Left': contract_left_days_input, 'Player Elapsed Time In Club': days_at_club_input, 'Your Club Name': club_name_input, 'Your Clubs League Name': league_name_input,  'Player Position': club_position_input,  'Player Nationality Name': nationality_name_input ,  'Player Preferred Foot': preferred_foot_input, 'Player Work Rate': work_rate_input, 'Player Body Type': body_type_input}

summary_df  = pd.DataFrame([summary_dictionary])
st.table(summary_df)




def predict_(model, value_eur_input, age_input, height_cm_input, weight_kg_input, league_level_input, days_at_club_input, preferred_foot_input, international_reputation_input, release_clause_eur_input, contract_left_days_input, club_name_input, league_name_input,club_position_input,nationality_name_input,work_rate_input,body_type_input):
    
    features = {'value_eur': value_eur_input,  'age': age_input, 'height_cm': height_cm_input,  'weight_kg': weight_kg_input, 'league_level': league_level_input, 'international_reputation': international_reputation_input, 'release_clause_eur': release_clause_eur_input,  'contract_left_days': contract_left_days_input, 'days_at_club': days_at_club_input, 'club_name': club_name_input, 'league_name': league_name_input,  'club_position': club_position_input,  'nationality_name':nationality_name_input ,  'preferred_foot': preferred_foot_input, 'work_rate': work_rate_input, 'body_type': body_type_input}

    features_df  = pd.DataFrame([features])
    
    prediction_ = model.predict(features_df, round = 0)
    prediction = int(prediction.Label[0])

    if prediction_ == 0:
        pred = 'red edildi.'
    else:
        pred = 'onaylandı.'
    
    return pred




st.markdown("---")

st.markdown("<h1 style='text-align: left; font-size: 20px;'>Girilen bilgilere göre başvuru sahibine kredi verilip verilmemesini öğrenmek için aşağıdaki butona tıklayınız:</h1>", unsafe_allow_html=True)

if st.button('Kredi verilsin mi?'):

    result_ = predict_(model, value_eur_input, age_input, height_cm_input, weight_kg_input, league_level_input, days_at_club_input, preferred_foot_input, international_reputation_input, release_clause_eur_input, contract_left_days_input, club_name_input, league_name_input,club_position_input,nationality_name_input,work_rate_input,body_type_input)

    if result_ == 'red edildi.':
        st.error('Krediniz {}'.format(result_))
    else:
        st.success('Krediniz {}'.format(result_))




# ---- STREAMLIT STİLİNİ SAKLA ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
