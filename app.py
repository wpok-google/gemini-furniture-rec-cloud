import os
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    return text_model_pro, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)


st.header("Gemini Furniture Recommendation", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

st.markdown(
    """In this demo, you will be presented with a scene (e.g., a living room) and will use the Gemini 1.0 Pro Vision model to perform visual understanding. You will see how Gemini 1.0 can be used to recommend an item (e.g., a chair) from a list of furniture options as input. You can use Gemini 1.0 Pro Vision to recommend a chair that would complement the given scene and will be provided with its rationale for such selections from the provided list.
            """
)

room_image_uri = (
    "gs://github-repo/img/gemini/retail-recommendations/rooms/living_room.jpeg"
)
chair_1_image_uri = (
    "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair1.jpeg"
)
chair_2_image_uri = (
    "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair2.jpeg"
)
chair_3_image_uri = (
    "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair3.jpeg"
)
chair_4_image_uri = (
    "gs://github-repo/img/gemini/retail-recommendations/furnitures/chair4.jpeg"
)

room_image_urls = (
    "https://storage.googleapis.com/" + room_image_uri.split("gs://")[1]
)
chair_1_image_urls = (
    "https://storage.googleapis.com/" + chair_1_image_uri.split("gs://")[1]
)
chair_2_image_urls = (
    "https://storage.googleapis.com/" + chair_2_image_uri.split("gs://")[1]
)
chair_3_image_urls = (
    "https://storage.googleapis.com/" + chair_3_image_uri.split("gs://")[1]
)
chair_4_image_urls = (
    "https://storage.googleapis.com/" + chair_4_image_uri.split("gs://")[1]
)

room_image = Part.from_uri(room_image_uri, mime_type="image/jpeg")
chair_1_image = Part.from_uri(chair_1_image_uri, mime_type="image/jpeg")
chair_2_image = Part.from_uri(chair_2_image_uri, mime_type="image/jpeg")
chair_3_image = Part.from_uri(chair_3_image_uri, mime_type="image/jpeg")
chair_4_image = Part.from_uri(chair_4_image_uri, mime_type="image/jpeg")

st.image(room_image_urls, width=350, caption="Image of a living room")
st.image(
    [
        chair_1_image_urls,
        chair_2_image_urls,
        chair_3_image_urls,
        chair_4_image_urls,
    ],
    width=200,
    caption=["Chair 1", "Chair 2", "Chair 3", "Chair 4"],
)

st.write(
    "Our expectation: Recommend a chair that would complement the given image of a living room."
)
content = [
    "Consider the following chairs:",
    "chair 1:",
    chair_1_image,
    "chair 2:",
    chair_2_image,
    "chair 3:",
    chair_3_image,
    "and",
    "chair 4:",
    chair_4_image,
    "\n"
    "For each chair, explain why it would be suitable or not suitable for the following room:",
    room_image,
    "Only recommend for the room provided and not other rooms. Provide your recommendation in a table format with chair name and reason as columns.",
]

tab1, tab2 = st.tabs(["Response", "Prompt"])
generate_image_description = st.button(
    "Generate recommendation....", key="generate_image_description"
)
with tab1:
    if generate_image_description and content:
        with st.spinner(
            "Generating recommendation using Gemini 1.0 Pro Vision ..."
        ):
            response = get_gemini_pro_vision_response(
                multimodal_model_pro, content
            )
            st.markdown(response)
with tab2:
    st.write("Prompt used:")
    st.text(content)
