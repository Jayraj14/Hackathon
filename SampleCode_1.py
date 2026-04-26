######### ---------- Please do not modify this code here. ---------- ##########
#### Download a copy of this code to your local machine to modify & test.  ####
####-----------------------------------------------------------------------####

#pip install langchain-openai 
from langchain_openai import ChatOpenAI  
import os  
import httpx  

client = httpx.Client(verify=False) 

llm = ChatOpenAI( 
    openai_api_base="https://genailab.tcs.in",
    model = "azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="sk-Your-API-KEY",
    # Will be provided during event.  And this key is for Hackathon purposes only
    # and should not be used for any unauthorized purposes 
    http_client = client 
) 

response = llm.invoke("Hi, i am a 30 year male") 
print(response)
