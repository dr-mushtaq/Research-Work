import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
from PIL import Image

# Load the fine-tuned T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Function to generate summary
def generate_summary(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate important points
def generate_points(text, num_sentences=3):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:num_sentences])

# Streamlit app
def main():
    # Custom CSS Styling
    st.markdown(
        """
        <style>
        body, .main, .block-container {
            background-color: #0000ff;
            color: #ffff00;
        }

        .big-title {
            font-size: 50px !important;
            color: #ffff00;
            font-weight: 900;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }

        textarea {
            background-color: #e0e0e0 !important;
            color: #0000ff !important;
            font-size: 22px !important;
            font-weight: bold !important;
        }

        .custom-label {
            font-size: 26px;
            font-weight: bold;
            color: #ffff00;
            margin-bottom: 10px;
        }

        .custom-subheader {
            font-size: 30px;
            font-weight: 900;
            color: #ffff00;
            margin-top: 30px;
        }

        .highlighted {
            background-color: #ffff00;
            color: #0000ff;
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }

        .result-section {
            background-color: #ffff00;
            color: #0000ff;
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Button styling */
        div.stButton > button:first-child {
            background-color: #ff0000;
            color: #ffff00;
            font-size: 26px;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 10px;
            border: 2px solid #ffff00;
            margin-top: 20px;
        }

        div.stButton > button:hover {
            background-color: #cc0000;
            color: #ffffff;
            border: 2px solid #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown('<div class="big-title">Text Summarizer and Important Points Generator</div>', unsafe_allow_html=True)

    # Display images
    image1 = Image.open("D:/image1.jpg")
    image2 = Image.open("D:/image2.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption='Image 1', use_column_width=True)
    with col2:
        st.image(image2, caption='Image 2', use_column_width=True)

    # Text input label
    st.markdown('<div class="custom-label">Enter your text here:</div>', unsafe_allow_html=True)
    user_input = st.text_area("", height=250)

    # Button
    if st.button("Generate Summary and Important Points"):
        if user_input:
            generated_summary = generate_summary(user_input)
            generated_points = generate_points(user_input)

            st.markdown('<div class="custom-subheader">Generated Summary:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-section">{generated_summary}</div>', unsafe_allow_html=True)

            st.markdown('<div class="custom-subheader">Important Points:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="highlighted">{generated_points}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before generating the summary and important points.")

if __name__ == "__main__":
    main()
