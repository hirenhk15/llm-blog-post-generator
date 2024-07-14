# LLM App to generate blog post from an user given topic.

import streamlit as st
from enum import Enum
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate


CREATIVITY=0.7
TEMPLATE = """As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
"""


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


# Defining prompt template
class FinalPromptTemplate:
    def __init__(self, topic:str) -> None:
        self.topic=topic
        
    def generate(self) -> str:
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=TEMPLATE
        )
        final_prompt = prompt.format(
            topic=self.topic
        )

        return final_prompt


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e
        

class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.sidebar.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.sidebar.success("Received valid API Key!")
            else:
                st.sidebar.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.sidebar.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="Blog Post Generator")
            st.markdown("<h1 style='text-align: center;'>Blog Post Generator</h1>", unsafe_allow_html=True)
            # st.title("Blog Post Generator")

            # Select the model provider
            option_model_provider = st.sidebar.selectbox(
                    'Model Provider',
                    ('GroqCloud', 'OpenAI')
                )
            
            # Input API Key for model to query
            api_key = self.get_api_key()

            # Get the topic from user
            topic_text = st.text_input("Enter topic: ")

            if topic_text:
                # Generate the final prompt
                final_prompt = FinalPromptTemplate(topic_text)
                print("Final Prompt: ", final_prompt.generate())
                
                # Load the LLM model
                llm_model = LLMModel(model_provider=option_model_provider)
                llm = llm_model.load(api_key=api_key)

                # Invoke the LLM model
                response = llm.invoke(final_prompt.generate(), max_tokens=2048)
                st.write(response.content)
        except Exception as e:
            st.error(str(e), icon=":material/error:")


def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()