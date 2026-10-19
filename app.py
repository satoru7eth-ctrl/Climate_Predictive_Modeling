import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("محاكي التنبؤ بالتغير المناخي 🌍")
st.write("محاكاة بسيطة لدرجات الحرارة المستقبلية باستخدام الانحدار الخطي")

# البيانات
data = {
    "year": [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020],
    "temperature": [28,28.5,29,29.3,29.7,30,30.4,30.8,31,31.3,31.7]
}

df = pd.DataFrame(data)

# نموذج الانحدار
X = df[["year"]]
y = df["temperature"]
model = LinearRegression()
model.fit(X, y)

# إدخال سنة مستقبلية
future_year = st.slider("اختاري سنة للتنبؤ:", 2021, 2050, 2025)
prediction = model.predict([[future_year]])

st.write(f"درجة الحرارة المتوقعة في سنة {future_year}: *{round(prediction[0], 2)}°C*")

# الرسم البياني
fig, ax = plt.subplots()
ax.scatter(df["year"], df["temperature"], color="blue", label="البيانات الحقيقية")
ax.plot(df["year"], model.predict(X), color="red", label="خط الاتجاه")
ax.set_xlabel("السنة")
ax.set_ylabel("درجة الحرارة")
ax.set_title("اتجاه ارتفاع درجات الحرارة")
ax.legend()
st.pyplot(fig)
