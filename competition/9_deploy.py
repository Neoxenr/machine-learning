import streamlit as st
import altair as alt
import pickle as pk
import pandas as pd
import numpy as np

def main():
    model = load_model()
    data = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Главная страница", "Модель"])
    if page == "Главная страница":
        st.title("House Prices – Advanced Regression Techniques")
        st.write("Каждый покупатель, описывая свой дом мечты, вероятно, не начнет с высоты подвального потолка или близости к железной дороге. Однако, данные этого соревнования доказывают, что на  окончательную цену недвижимости это влияет гораздо больше, чем количество спален или забор из белого штакетника. Соревнование ставит задачу предсказать окончательную цену каждого дома, расположенного в Эймсе, штате Айова.")
        st.image("housesbanner.png")
        st.write("Набор данных содержит 79 признаков, которые почти полностью описывают каждый аспект жилых домов в Эймсе, штате Айова. Данный набор данных был составлен профессором Де Коком для использования в образовании в области науки о данных. Это невероятная альтернатива для исследователей данных, ищущих модернизированную и расширенную версию часто используемого набора данных о жилье в Бостоне.")
    elif page == "Модель":
        st.title("Взаимодействие с моделью")
        action = st.selectbox("Выберите действие", ["Вывести информацию о доме", "Предсказать стоимость дома"])
        if action == "Вывести информацию о доме":
            value = st.number_input("Введите ID дома", step=1)
            st.write(data[value: value + 1])
        elif action == "Предсказать стоимость дома":
            value = st.number_input("Введите ID дома", step=1)
            st.write(predict(model, data[value:value + 1]))
        
@st.cache
def load_model():
    with open('ridge_model.pkl', 'rb') as pkl_file:
        ridge_model = pk.load(pkl_file)
    return ridge_model

@st.cache
def load_data():
    return pd.read_csv("../data/house_prices_preprocessed.csv")

@st.cache
def predict(model, X):
    return np.exp(model.predict(X))
        
if __name__ == "__main__":
    main()