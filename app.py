
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="centered"
)

# Class names must remain in the same order used during training
CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and related lesions",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesions"
}

# The model file must be in the same folder as app.py
MODEL_PATH = "skin_lesion_classification_final.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🔬 Skin Lesion Classification")
st.write(
    "Upload a skin image to see the class predicted by "
    "the deep learning model."
)

st.warning(
    "Educational use only. This application is not a medical "
    "diagnostic tool. Model predictions may be incorrect. "
    "Consult a qualified healthcare professional for medical concerns."
)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded skin image",
            use_container_width=True
        )

        if st.button("Classify Image"):
            with st.spinner("Analyzing image..."):
                model = load_model()

                # Preprocess the image
                resized_image = image.resize((224, 224))
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                # Generate predictions
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                predicted_score = float(predictions[predicted_index])

            st.subheader("Prediction Result")

            st.write(f"**Predicted class:** {predicted_class}")
            st.write(
                f"**Description:** "
                f"{CLASS_DESCRIPTIONS[predicted_class]}"
            )
            st.write(
                f"**Highest model score:** "
                f"{predicted_score * 100:.2f}%"
            )

            st.caption(
                "The model score is not a measure of diagnostic certainty."
            )

            st.subheader("Scores for all classes")

            for class_name, probability in zip(
                CLASS_NAMES, predictions
            ):
                st.write(
                    f"{class_name}: {float(probability) * 100:.2f}%"
                )
                st.progress(float(probability))

    except Exception as error:
        st.error(
            "Unable to process the image or load the model. "
            "Please check the model file and try again."
        )
        st.error(str(error))
