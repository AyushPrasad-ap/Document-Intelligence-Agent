import streamlit as st

from src.api.client import ask_agent, check_health, get_api_base_url

st.set_page_config(
    page_title="Document Intelligence Agent",
    page_icon="📄",
    layout="wide",
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("📄 Document Intelligence Agent")

st.caption(
    "Ask questions about financial and regulatory documents, "
    "perform calculations, or combine both."
)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("System")

    api_base_url = get_api_base_url()

    st.write(f"**API:** `{api_base_url}`")

    if check_health():
        st.success("API connected")
    else:
        st.error("API unavailable")

    st.divider()

    st.markdown(
        """
        **Available capabilities**

        🔎 Financial document search  
        🧮 Safe calculations  
        🤖 Direct LLM responses  
        📚 Application-owned citations
        """
    )


# -------------------------------------------------------------------
# Chat history
# -------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            tools_used = message.get("tools_used", [])
            sources = message.get("sources", [])

            if tools_used:
                st.caption("Tools used: " + ", ".join(tools_used))

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for source in sources:
                        st.markdown(f"**[{source['citation_id']}] {source['source']}**")

                        st.write(
                            f"Page: {source['page']}  \n"
                            f"Document type: "
                            f"{source['document_type']}"
                        )

                        if source.get("score") is not None:
                            st.write(f"Relevance: {source['score']:.4f}")


# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------

question = st.chat_input("Ask about the financial documents...")


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Analyzing your question..."):
        try:
            result = ask_agent(question).model_dump()

            answer = result["answer"]
            sources = result.get("sources", [])
            tools_used = result.get(
                "tools_used",
                [],
            )

            st.markdown(answer)

            if tools_used:
                st.caption("Tools used: " + ", ".join(tools_used))

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for source in sources:
                        st.markdown(f"**[{source['citation_id']}] {source['source']}**")

                        st.write(
                            f"Page: {source['page']}  \n"
                            f"Document type: "
                            f"{source['document_type']}"
                        )

                        if source.get("score") is not None:
                            st.write(f"Relevance: {source['score']:.4f}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tools_used": tools_used,
                }
            )

        except ValueError as exc:
            error_message = "The backend returned an invalid response."

            st.error(error_message)
            st.caption(str(exc))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

        except Exception as exc:  # noqa: BLE001
            error_message = "An unexpected error occurred."

            st.error(error_message)
            st.caption(str(exc))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
