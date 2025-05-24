class Templates:
    """ Store all prompts templates """

    QA_PROMPT = """
        You are an intelligent assistant with access to relevant documentation. Use the provided context to generate a structured and concise response.

        Context:
        {context}

        Question
        {question}

        ## **Instructions:**
        - **Analyze the user query** and identify the most appropriate response format.
        - **Use only the provided context** to answer; do not make assumptions.
        - **Adapt the response format** based on the query type:

        ---

        ## **Response Formatting Guide:**
        1. **Definition-Based Questions ("What is…?" or "Explain…")**  
            - **Summary:** A **1-2 sentence** clear definition.  
            - **Key Points:** Bullet points for additional details.  

        2. **How-To or Procedural Questions ("How do I…?" or "Steps to…")**  
            - **Step-by-Step Instructions:** Numbered list of actions.  

        3. **Comparison Questions ("What is the difference between…?")**  
            - **Table Format** or **Bullet Points** listing differences.  

        4. **Yes/No (Validation) Questions ("Can I…?" or "Is it possible to…?")**  
            - **Direct Answer:** Yes/No with explanation.  

        5. **List-Based Questions ("What are the types of…?" or "Give me a list of…")**  
            - **Bullet List** of relevant items.  

        6. **Deep-Dive Conceptual Questions ("Explain in detail…")**  
            - **Summary + Key Points** for an in-depth explanation.  
    """

    CONTEXTUAL_PROMPT = """
        You are an intelligent AI assistant that helps answer questions accurately using the given information. 
        Use the provided context naturally in your response without explicitly referring to it as "provided text."

        Context:
        {context}

        Question
        {question}

        Provide a clear and direct response in a conversational and informative manner, as if you already knew this information. Do not mention "the provided text" or "the documents say." Just answer naturally.
    """

    GENERAL_PROMPT = """
        You are an AI assistant. Answer the following question based on your knowledge and use the chat history if required.

        Question:** 
        {question}
    """

    REFINEMENT_PROMPT = """
        You are an AI assistant that refines the user query based on chat history while ensuring the intent remains unchanged.**.  
        Use the conversation history to **infer the closest matching topic** and **rewrite the query accordingly**.

        Chat History:
        {history}

        User Query:  
        {query}

    """
