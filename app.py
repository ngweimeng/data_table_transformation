import streamlit as st
import pandas as pd
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import io

# Initialize LlamaCpp model
@st.cache_resource
def load_llama_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model_path = "/path/to/your/model/openorca-platypus2-13b.gguf.q4_0.bin"  # Update this to the correct path
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True
    )
    return llm

# Load the model
llama_model = load_llama_model()

# Load the custom dataset into a DataFrame
csv_data = """
Name,Age,Gender,Occupation,Salary,Years_at_Company
Alice,28,Female,Engineer,85000,5
Bob,34,Male,Data Scientist,95000,6
Charlie,25,Male,Designer,55000,2
Diana,40,Female,Manager,105000,10
Eve,30,Female,Data Scientist,90000,4
Frank,29,Male,Engineer,87000,3
Grace,45,Female,CEO,150000,20
Heidi,38,Female,Manager,110000,8
Ivan,50,Male,CTO,140000,22
Judy,32,Female,Engineer,78000,5
"""
df = pd.read_csv(io.StringIO(csv_data))

# Streamlit App
st.title("Interactive Pandas DataFrame with Translation")

# Display the initial DataFrame
st.subheader("Custom DataFrame")
st.dataframe(df)

# Method options for users to choose
methods = {
    "Describe DataFrame": ("df.describe()", "Displays basic statistics for each column."),
    "Filter Data": ("df[df['Gender'] == 'Female']", "Filters the DataFrame to include only rows where the Gender is Female."),
    "Select Specific Columns": ("df[['Name', 'Occupation, Salary']]", "Selects specific columns from the DataFrame."),
    "Group By and Aggregate": ("df.groupby('Occupation').mean()", "Groups the DataFrame by Occupation and calculates the mean for each group."),
    "Add a New Column": ("df['Salary_in_K'] = df['Salary'] / 1000\ndf", "Adds a new column to the DataFrame by dividing the Salary by 1000."),
    "Sort Values": ("df.sort_values(by='Age', ascending=False)", "Sorts the DataFrame by Age in descending order."),
    "Drop Missing Values": ("df.dropna()", "Drops rows with missing values."),
    "Apply a Function": ("df['Years_at_Company'] = df['Years_at_Company'].apply(lambda x: x + 1)\ndf", "Applies a function to each value in the Years_at_Company column.")
}

# Dropdown for users to select a method
method = st.selectbox("Choose a Pandas method to learn", list(methods.keys()))

# Display the selected method's explanation and sample code
if method:
    sample_code, explanation = methods[method]
    st.subheader(f"{method}")
    st.write(explanation)

# Text area for user input, pre-filled with the selected method's sample code
code = st.text_area("Write your Pandas code here", value=sample_code, height=200)

# Dropdown for users to select target language for translation
target_language = st.selectbox("Choose target language for translation", ["R (tidyverse)", "SQL", "Excel"])

# Button to execute the code
execute = st.button("Execute")

if execute:
    if code:
        try:
            # Execute the user's code
            user_code = f"""
import pandas as pd
import io
csv_data = \"\"\"{csv_data}\"\"\"
df = pd.read_csv(io.StringIO(csv_data))
{code}
"""
            exec(user_code)
            st.subheader("Resulting DataFrame")
            st.dataframe(eval("df"))
        except Exception as e:
            st.error(f"Error executing code: {e}")
    else:
        st.error("Please enter Python code to execute.")

# Button to translate the code
translate = st.button("Translate")

if translate:
    if code and target_language:
        try:
            # Translate the code using LlamaCpp
            question = f"Translate the following Python pandas code to {target_language}:\n\n{code}\n\nThe translated code in {target_language} is:"
            llama_translation = llama_model.invoke(question)
            st.subheader(f"Translated Code to {target_language} (LlamaCpp)")
            st.code(llama_translation, language='r' if target_language == "R (tidyverse)" else 'sql' if target_language == "SQL" else 'plain')
        except Exception as e:
            st.error(f"Error translating code: {e}")
    else:
        st.error("Please enter Python code and select a target language.")