import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd

#session = get_active_session()

cnx = st.connection("snowflake")
session = cnx.session()

pd.set_option("max_colwidth", None)
num_chunks = 3

def get_accessible_documents():
    role = session.sql("select current_role() as role").collect()[0]["ROLE"]
    
    # Query to check the access control table and get accessible documents
    access_query = """
        SELECT PDF_ACCESS
        FROM CC_QUICKSTART_CORTEX_DOCS.DATA.DOCS_ACCESS_CONTROL
        WHERE ROLE = ?
    """
    access_results = session.sql(access_query, params=[role]).to_pandas()

    if 'ALL' in access_results['PDF_ACCESS'].values:
        # If role has access to all documents, fetch all document paths
        docs_available = session.sql("ls @docs").collect()
    else:
        # Fetch only the documents the role has access to
        accessible_docs_set = set(access_results['PDF_ACCESS'].values)
        # Extract document names that the role can access
        docs_available = session.sql("ls @docs").collect()
        accessible_docs_set = {doc.split('/')[-1] for doc in accessible_docs_set}  # Extract filenames only
        docs_available = [doc for doc in docs_available if doc["name"] in accessible_docs_set]
    
    return docs_available

def create_prompt(myquestion, rag):
    if rag == 1:
        cmd = """
        with results as
        (SELECT RELATIVE_PATH,
           VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', ?)) as similarity,
           chunk
        from docs_chunks_table
        order by similarity desc
        limit ?)
        select chunk, relative_path from results 
        """
    
        df_context = session.sql(cmd, params=[myquestion, num_chunks]).to_pandas()      
        
        context_length = len(df_context) - 1

        prompt_context = ""
        for i in range(0, context_length):
            prompt_context += df_context._get_value(i, 'CHUNK')

        prompt_context = prompt_context.replace("'", "")
        relative_path =  df_context._get_value(0, 'RELATIVE_PATH')
    
        prompt = f"""
          'You are an expert assistant extracting information from context provided. 
           Answer the question based on the context. Be concise and do not hallucinate. 
           If you don’t have the information just say so.
          Context: {prompt_context}
          Question:  
           {myquestion} 
           Answer: '
           """
        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{relative_path}', 360) as URL_LINK from directory(@docs)"
        df_url_link = session.sql(cmd2).to_pandas()
        url_link = df_url_link._get_value(0, 'URL_LINK')

    else:
        prompt = f"""
         'Question:  
           {myquestion} 
           Answer: '
           """
        url_link = "None"
        relative_path = "None"
        
    return prompt, url_link, relative_path

def complete(myquestion, model_name, rag=1):
    prompt, url_link, relative_path = create_prompt(myquestion, rag)
    cmd = f"""
             select SNOWFLAKE.CORTEX.COMPLETE(?,?) as response
           """
    
    df_response = session.sql(cmd, params=[model_name, prompt]).collect()
    return df_response, url_link, relative_path

def display_response(question, model, rag=0):
    response, url_link, relative_path = complete(question, model, rag)
    res_text = response[0].RESPONSE
    st.markdown(res_text)
    if rag == 1 and relative_path != "None":
        display_url = f"Link to [{relative_path}]({url_link}) that may be useful"
        st.markdown(display_url)

# Main code
st.title("Asking Questions to Your Own Documents with Snowflake Cortex:")
st.write("""You can ask questions and decide if you want to use your documents for context or allow the model to create its own response.""")

docs_available = get_accessible_documents()
list_docs = [doc["name"] for doc in docs_available]
st.dataframe(list_docs)

model = st.sidebar.selectbox('Select your model:', (
                                    'mixtral-8x7b',
                                    'snowflake-arctic',
                                    'mistral-large',
                                    'llama3-8b',
                                    'llama3-70b',
                                    'reka-flash',
                                    'mistral-7b',
                                    'llama2-70b-chat',
                                    'gemma-7b'))

question = st.text_input("Enter question", placeholder="Ask Questions", label_visibility="collapsed")

rag = st.sidebar.checkbox('Use your own documents as context?')

if rag:
    use_rag = 1
else:
    use_rag = 0

if question:
    display_response(question, model, use_rag)
