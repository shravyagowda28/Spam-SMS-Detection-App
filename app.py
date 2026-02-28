import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam SMS Detection App")
st.write("Enter a message to check whether it is Spam or Ham.")

user_input = st.text_area("Type your message here:")

if st.button("Check"):
    if user_input:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)
        
        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
    else:
        st.warning("Please enter a message.")
