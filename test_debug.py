import streamlit as st
import ollama
import chromadb
import pandas as pd

def initialize():
    # 检查 'already_executed' 状态，确保不重复初始化数据库
    if not st.session_state.get('already_executed', False):
        setup_database()  # 调用设置数据库的函数
        st.session_state['already_executed'] = True  # 设置状态为 True，表示已初始化


def setup_database():
    client = chromadb.Client()
    file_path = 'QA_pipi.xlsx'
    documents = pd.read_excel(file_path, header=None)
    collection = client.get_or_create_collection(name="demodocs")

    for index, content in documents.iterrows():
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content[0])
        collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]])

    st.session_state['already_executed'] = True
    st.session_state['collection'] = collection
    #st.write("已将 already_executed 设置为 True")

def main():
    st.title("LLM+RAG test")
    initialize()
    user_input = st.text_area("您想問什麼？", "")
    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)
        else:
            st.warning("請輸入問題！")

def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)
    data = results['documents'][0]
    output = ollama.generate(
        model="llama3",
        #model="llama3/Meta-Llama-3-8B-Instruct",
        prompt=f"Using this data: {data}. Respond to this prompt and use Chinese: {user_input}"
    )
    st.text("回答：")
    st.write(output['response'])

if __name__ == "__main__":
    main()

