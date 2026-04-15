# Step by step tutorial for Legal Eagles Codelab
refers to codelab: https://codelabs.developers.google.com/legal-eagle-rag/instructions#0

### Changes from the original lab
- Changes on Dockerfile to use UV
- Changes on the deployment do Cloud Registry

### Products and services used
https://console.cloud.google.com/firestore
https://console.cloud.google.com/storage
https://console.cloud.google.com/artifacts
https://console.cloud.google.com/run
https://console.cloud.google.com/eventarc


## 1 (Introduction) - 2 (Architecture) - 3 (Before you begin)

Stands for stages [1](https://codelabs.developers.google.com/legal-eagle-rag/instructions#0), [2](https://codelabs.developers.google.com/legal-eagle-rag/instructions#1) and [3](https://codelabs.developers.google.com/legal-eagle-rag/instructions#2).

```
gcloud auth list
```

```
gcloud config set project <YOUR_PROJECT_ID>
``` 

```
gcloud services enable storage.googleapis.com  \
                        run.googleapis.com  \
                        artifactregistry.googleapis.com  \
                        aiplatform.googleapis.com \
                        eventarc.googleapis.com \
                        cloudresourcemanager.googleapis.com \
                        firestore.googleapis.com \
                        cloudaicompanion.googleapis.com
```

```
git clone https://github.com/weimeilin79/legal-eagle.git
```

## 4 (Writing the Inference Application with Gemini Code Assist) - 5 (Local Testing in Cloud Editor)

Stands for stages [4](https://codelabs.developers.google.com/legal-eagle-rag/instructions#3) and [5](https://codelabs.developers.google.com/legal-eagle-rag/instructions#4) 

On legal.py replace:
```python
import os
import signal
import sys
import vertexai
import random
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
# Connect to resourse needed from Google Cloud
llm = VertexAI(model_name="gemini-2.5-flash")
def ask_llm(query):
    try:
        query_message = {
            "type": "text",
            "text": query,
        }

        input_msg = HumanMessage(content=[query_message])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant, and you are with the attorney in a courtroom, you are helping him to win the case by providing the information he needs "
                        "Don't answer if you don't know the answer, just say sorry in a funny way possible"
                        "Use high engergy tone, don't use more than 100 words to answer"
                       # f"Here is some past conversation history between you and the user {relevant_history}"
                       # f"Here is some context that is relevant to the question {relevant_resource} that you might use"
                    )
                ),
                input_msg,
            ]
        )
        prompt = prompt_template.format()
        response = llm.invoke(prompt)
        print(f"response: {response}")
        return response
    except Exception as e:
        print(f"Error sending message to chatbot: {e}") # Log this error too!
        return f"Unable to process your request at this time. Due to the following reason: {str(e)}"
```

main.py before `__name__ == "__main__":`
```python
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    try:
        # call the ask_llm in legal.py
        answer_markdown = legal.ask_llm(question)
        
        print(f"answer_markdown: {answer_markdown}")
        # Return the Markdown as the response
        return answer_markdown, 200
    except Exception as e:
        return f"Error: {str(e)}", 500  # Handle errors appropriately
```

create a .gitignore file containing
```python
.venv
__pycache__
```

Testing
```bash
cd ~/legal-eagle/webapp
python -m venv .venv
source .venv/bin/activate
export PROJECT_ID=$(gcloud config get project)
uv pip install -r requirements.txt
python main.py
```

Exit the environment
```bash
deactivate
```

## 6. Setting up the Vector Store


Follow the visual steps on https://codelabs.developers.google.com/legal-eagle-rag/instructions#5 

Creating the vector index
```bash
export PROJECT_ID=$(gcloud config get project)
gcloud firestore indexes composite create \
--collection-group=legal_documents \
--query-scope=COLLECTION \
--field-config field-path=embedding,vector-config='{"dimension":"768", "flat": "{}"}' \
--project=${PROJECT_ID}
```

## 7. Loading Data into the Vector Store

Follow the visual steps on https://codelabs.developers.google.com/legal-eagle-rag/instructions#6

## 8. Set up a Cloud Run Function

This stands for https://codelabs.developers.google.com/legal-eagle-rag/instructions#7

```bash
cd ~/legal-eagle
mkdir loader
cd loader
```

create the files
```bash
touch main.py requirements.txt Dockerfile .gitignore
```

paste this in main.py
```python
import os
import json
from google.cloud import storage
import functions_framework
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import vertexai
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")  # Get project ID from env
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004" ,
    project=PROJECT_ID,)
COLLECTION_NAME = "legal_documents"
# Create a vector store
vector_store = FirestoreVectorStore(
    collection="legal_documents",
    embedding_service=embedding_model,
    content_field="original_text",
    embedding_field="embedding",
)
@functions_framework.cloud_event
def process_file(cloud_event):
    print(f"CloudEvent received: {cloud_event.data}")  # Print the parsed event data
     
    """Triggered by a Cloud Storage event.
       Args:
            cloud_event (functions_framework.CloudEvent): The CloudEvent
                containing the Cloud Storage event data.
    """
    try:
        event_data = cloud_event.data
        bucket_name = event_data['bucket']
        file_name = event_data['name']
    except (json.JSONDecodeError, AttributeError, KeyError) as e:  # Catch JSON errors
        print(f"Error decoding CloudEvent data: {e} - Data: {cloud_event.data}")
        return "Error processing event", 500  # Return an error response
   
    print(f"New file detected in bucket: {bucket_name}, file: {file_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    try:
        # Download the file content as string (assuming UTF-8 encoded text file)
        file_content_string = blob.download_as_string().decode("utf-8")
        print(f"File content downloaded. Processing...")
        # Split text into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        text_chunks = text_splitter.split_text(file_content_string)
        print(f"Text split into {len(text_chunks)} chunks.")
 
        # Add the docs to the vector store
        vector_store.add_texts(text_chunks)    
        print(f"File processing and Firestore upsert complete for file: {file_name}")
        return "File processed successfully", 200  #  Return success response
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
```

now paste this in the requirements
```python
Flask==2.3.3
requests==2.31.0
google-generativeai>=0.2.0
langchain
langchain_google_vertexai
langchain-community
langchain-google-firestore
google-cloud-storage
functions-framework
```

and paste this in .gitignore
```
.venv
__pycache__
```

## 9. Test and build Cloud Run Function

This stands for https://codelabs.developers.google.com/legal-eagle-rag/instructions#8

```bash 
cd ~/legal-eagle/loader
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Lets emulate
```bash
functions-framework --target process_file --signature-type=cloudevent --source main.py
```

In a new terminal, lets upload one case (txt) into our bucket
```bash
export DOC_BUCKET_NAME=$(gcloud storage buckets list --format="value(name)" | grep doc-bucket)
gsutil cp ~/legal-eagle/court_cases/case-01.txt gs://$DOC_BUCKET_NAME/
```

Now lets trigger locally our cloud function code to make and store the embeddings
```bash
curl -X POST -H "Content-Type: application/json" \
     -d "{
       \"specversion\": \"1.0\",
       \"type\": \"google.cloud.storage.object.v1.finalized\",
       \"source\": \"//storage.googleapis.com/$DOC_BUCKET_NAME\",
       \"subject\": \"objects/case-01.txt\",
       \"id\": \"my-event-id\",
       \"time\": \"2024-01-01T12:00:00Z\",
       \"data\": {
         \"bucket\": \"$DOC_BUCKET_NAME\",
         \"name\": \"case-01.txt\"
       }
     }" http://localhost:8080/
```

Lets exit our venv
```bash
deactivate
```

## 10. Build container image and push to Artifacts repositories

This stands for https://codelabs.developers.google.com/legal-eagle-rag/instructions#9 

Paste this on our dockerfile
```Dockerfile
# Use a Python 3.12 slim base image
FROM python:3.12-slim

# Adiciona o uv copiando o binário oficial (método mais rápido)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt and install Python dependencies using uv
COPY requirements.txt .

# Instala as dependências com o uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy main.py
COPY main.py .

# Set the command to run functions-framework
CMD ["functions-framework", "--target", "process_file", "--port", "8080"]
```

create the artifact to store our docker image
```bash
gcloud artifacts repositories create legal-eagles-repository \
    --repository-format=docker \
    --location=us-central1 \
    --description="legal-eagles-repository"
```

upload to model registry
```bash
export PROJECT_ID=$(gcloud config get project)
gcloud builds submit --tag us-central1-docker.pkg.dev/${PROJECT_ID}/legal-eagles-repository/legal-eagle-loader:latest . 
```

## 11. Create the Cloud Run function and set up Eventarc trigger

Follow the steps on https://codelabs.developers.google.com/legal-eagle-rag/instructions#10

## 12. Upload legal documents to the GCS bucket

This stands for https://codelabs.developers.google.com/legal-eagle-rag/instructions#11 

Lets upload more cases into the vector DB/Firestore
```bash
export DOC_BUCKET_NAME=$(gcloud storage buckets list --format="value(name)" | grep doc-bucket)
gsutil cp ~/legal-eagle/court_cases/case-02.txt gs://$DOC_BUCKET_NAME/
gsutil cp ~/legal-eagle/court_cases/case-03.txt gs://$DOC_BUCKET_NAME/
gsutil cp ~/legal-eagle/court_cases/case-06.txt gs://$DOC_BUCKET_NAME/
```

## 12. Implementing RAG

Add the imports in legal.py
```python 
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore
```

```python
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
COLLECTION_NAME = "legal_documents"

embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=PROJECT_ID,
)

vector_store = FirestoreVectorStore(
    collection=COLLECTION_NAME,
    embedding_service=embedding_model,
    content_field="original_text",
    embedding_field="embedding",
)
```

Lets paste our helper function
```python
def search_resource(query):
    results = []
    results = vector_store.similarity_search(query, k=5)
    
    combined_results = "\n".join([result.page_content for result in results])
    print(f"==>{combined_results}")
    return combined_results
```

Lets replace the ask_llm function
```python 
def ask_llm(query):
    try:
        query_message = {
            "type": "text",
            "text": query,
        }
        relevant_resource = search_resource(query)
       
        input_msg = HumanMessage(content=[query_message])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant, and you are with the attorney in a courtroom, you are helping him to win the case by providing the information he needs "
                        "Don't answer if you don't know the answer, just say sorry in a funny way possible"
                        "Use high engergy tone, don't use more than 100 words to answer"
                        f"Here is some context that is relevant to the question {relevant_resource} that you might use"
                    )
                ),
                input_msg,
            ]
        )
        prompt = prompt_template.format()
        
        response = llm.invoke(prompt)
        print(f"response: {response}")
        return response
    except Exception as e:
        print(f"Error sending message to chatbot: {e}") # Log this error too!
        return f"Unable to process your request at this time. Due to the following reason: {str(e)}"
```

And now, lets test-it
```bash
cd ~/legal-eagle/webapp
source .venv/bin/activate
python main.py
```

lets exit the environment
```bash
deactivate
```

# Bonus

Fofão Criminal Case
```txt
The People of the State of California v. Fofão (Carreta Furacão)

Charge: Involuntary Manslaughter and Reckless Parkour
Summary:
Background: Sundar Pichai, CEO of Google, was fatally injured during an outdoor tech demonstration when Fofão, a star member of the Brazilian street-dancing troupe "Carreta Furacão," executed an unauthorized, high-velocity backflip off a moving brightly lit truck. Fofão, known for his oversized cheeks and chaotic acrobatics, landed directly on Pichai, who was entirely distracted while attempting to optimize a new Google Maps algorithm for street performers. 
Legal Arguments: The prosecution argued that Fofão intentionally ignored pedestrian safety protocols, wielding his chaotic dance energy and heavy synthetic mask as a lethal blunt instrument. The defense claimed that Fofão's actions were standard procedure during the chorus of "Siga em Frente, Olhe para o Lado," and that Pichai assumed the risk by standing in a clearly active breakdancing and parkour zone.
Witness Statements: Fellow troupe members, wearing unlicensed Spider-Man and Popeye costumes, testified that Fofão was simply "vibing at a highly dangerous frequency" and could not abort the flip mid-air. A forensic biomechanic provided evidence of the tragic trajectory, confirming that Fofão's sheer momentum, combined with the heavy techno-funk bassline, made the impact mathematically unavoidable.
Verdict: The jury found Fofão guilty of Involuntary Manslaughter. He was sentenced to 15 years in federal prison without his magical dancing train, and ordered to pay restitution by permanently replacing the Google Assistant voice with his own terrifying laughter.
```

```bash
export DOC_BUCKET_NAME=$(gcloud storage buckets list --format="value(name)" | grep doc-bucket)
gsutil cp ~/legal-eagle/court_cases/case-fofao.txt gs://$DOC_BUCKET_NAME/
```

# Deploy

```bash
cd ~/legal-eagle/webapp
```

Change the docker file
```bash
# Python image to use.
FROM python:3.12.8-slim-bullseye

# Adiciona o uv copiando o binário oficial (método mais rápido)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN uv pip install --system --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python", "main.py"]
```

```bash
export PROJECT_ID=$(gcloud config get project)
gcloud builds submit --tag us-central1-docker.pkg.dev/${PROJECT_ID}/legal-eagles-repository/legal-eagle-webapp:latest . 
```

```bash
export PROJECT_ID=$(gcloud config get project)
gcloud run deploy legal-eagle-webapp \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/legal-eagles-repository/legal-eagle-webapp \
  --region us-central1 \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=${PROJECT_ID}  \
  --allow-unauthenticated
```
