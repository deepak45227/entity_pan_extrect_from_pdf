# AI-Powered PAN–Entity Extraction using LangChain + Gemini (and Groq Optional)

This project extracts PAN numbers and corresponding Names/Organisations from large, unstructured PDF documents such as SEBI orders, financial notices, and compliance reports.

Divide the data into 2 chunks take 3 attempts sending tokens gemini , pause in betwwen next attempt  , use diffrent models of gemini and final output csv 

It uses:

- Gemini AI (Google’s LLM)

- pypdf

- Structured LLM prompts

- Clean CSV export for analysis or assignment submission


## How to run 


## 📦 Install Dependencies

```bash
pip install -r requirements.txt

```


## GEmini API KEY 
``` paste your api key in .env file  ```

``` change your file path  accordingly  ```


## Run code 
```  python exteact.py ``` 

## final output 
 - result.csv