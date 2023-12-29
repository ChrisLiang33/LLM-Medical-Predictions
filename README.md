fine tune LLM project from cs149


				        CS 149 Project 2 report
Team members: Zhihan Li, Xianmai Liang, Wenhao Xie

In this project each of us have tried different combinations of prompts, truncation/summarizing methods, text embedding techniques, solutions for  imbalanced data, and base models to process the medical notes data and fine-tune a LLM for ARDS prediction purposes.

We first explored the available options for the things mentioned above, we experimented with 3 different prompts, 2 truncation/summarizing methods, 1 text embedding technique, 3 solutions for imbalance data and  2 types of base models. In total there are 3*2*1*3*1=18 different combinations and we split them equally among the three of us. It turns out that the model with original prompt provided by TA, splitting up the medical notes for the patients who have ARDS to meet the memory restriction and using llama-2 as base model gives the best performance in terms of f1 score and other accuracy measures.

[Problem caused the late submission]

At 11/10 in the evening, we encountered an unusual issue while attempting to load a pre-trained model from HuggingFace. In a nutshell, we can only make predictions only after we finish the fine tuning process. However, if we try to make predictions with our model loaded from hugging face we get an AttributeError: 'LlamaForCausalLM' object has no attribute 'predict' error.
We used the same code for prediction, but it works only after we finish the fine tuning process, so we need to wait for a 3 hours long fine tuning process every time we make a prediction. Consequently, we had to retrain the models in order to obtain the test results, which resulted in a delay of several hours in our project submission. All team members experienced the same error when trying to load their respective pre-trained models, leading us to suspect that this issue may originate from a bug within the model itself but we could not debug it.

[Chris Liang]
Truncation/summarizing methods by Chris Liang
The default truncation method which considers only the first 1260 characters of the medical notes wastes too much useful information. 
Method 1:
I tried several methods with summarizing the data. The approach was using summarization models on hugging face, specially the medical ones  including the Falconsai/medical_summarization on hugging face and several others. The issue with that is taking a long time up to half a minute per note, and it becomes unfeasible with 8000+ plus notes. Then I tried a different method. 
Method 2:
I tried to split positive notes into individual notes, so one True Note became upwards of 70 small notes. And for False Notes, we remain the same. This way, we expanded the entries of True notes to preserve useful information and tackle the imbalance data at the same time, the total entry became around 11800 and negative: positive ratio became 8000:3800

Below is the method I used to process the data
import pandas as pd


# Function to process each row based on the label
def process_row(row):
   if row['label']:
       # If label is True, split the notes and create new rows
       notes = row['text'].split("Note")
       # Remove the first empty element and trim spaces
       notes = [note.strip() for note in notes if note.strip()]
       # Create new rows for each note with updated index
       return [{'index': row['index'] + i, 'text': f"Note {note}", 'label': row['label'], 'split': row['split']} for i, note in enumerate(notes)]
   else:
       # If label is False, take only the first 50 words
       truncated_text = ' '.join(row['text'].split())
       return [{'index': row['index'], 'text': truncated_text, 'label': row['label'], 'split': row['split']}]


# Initializing an empty list to store processed rows
processed_rows = []


# Process each row and update the index accordingly
current_index = 0
for _, row in df.iterrows():
   row['index'] = current_index
   new_rows = process_row(row)
   processed_rows.extend(new_rows)
   current_index += len(new_rows)


# Creating a new DataFrame with the processed rows
df = pd.DataFrame(processed_rows)


[Wenhao Xie]
Prompts I used:
Given a patient's medical notes, please provide a prediction for the likelihood that that patient is having Acute Respiratory Distress Syndrome (ARDS). Please return ‘true’ if the likelihood is greater than or equal to 70% and return ‘false’ otherwise. 
Given the historical medical notes, please forecast the future course of Acute Respiratory Distress Syndrome (ARDS). Please return ‘true’ if the patient is likely to develop ARDS, return ‘false’ otherwise.

Summarization method and deal with imbalance data:
I used TF-IDF to convert text data into numerical features and applied SMOTE technique to oversample the minority class


Base model:
I tried to use several LLMs for token/text classification, for example medical-ner-proj/bert-medical-ner-proj which is a medical documents NER model by fine tuning BERT. However, I got unsolvable errors from using these models (it requires changing the source code of Ludwig) so I stuck with the original llama2 model.


[Zhihan Li]
Trained 3 models. First one is from the default setting with the change in temperature from 0 to 0.0001, not fine-tuning at all.. Second one is fine-tuning with extract keywords related to ARDS and a balanced dataset(very little, from 100 to 300 using resampling).

Below is the description of the fine-tuning model(second model).
Prompt: Using the default prompt from the colab instruction.
Truncation: Using the English model in spaCy, extract keywords(words related to ARDS) from the input.
# Load the English model from spacy. This model will be used for processing English text.
nlp = spacy.load("en_core_web_sm")
# Define a function to summarize medical notes using spacy.
def summarize_notes_spacy(note):
    # Process the note using the spacy model.
    doc = nlp(note)
    # Define a list of keywords related to respiratory conditions.
    keywords = ["ARDS", "pneumonia", "edema", "pleural effusion", "breathlessness", "lung injury", "hypoxemia", "hypoxia", "respiratory failure","ventilator", "mechanical ventilation", "intubation","pulmonary infiltrates", "sepsis", "inflammation", "fibrosis","acute lung injury"]
    # Initialize an empty list to hold sentences that contain the keywords.
    summary_sentences = []
    # Iterate through each sentence in the processed document.
    for sent in doc.sents:
        # Check if any of the keywords are in the sentence.
        if any(keyword.lower() in sent.text.lower() for keyword in keywords):
            # If a keyword is found, add the sentence to the summary list.
            summary_sentences.append(sent.text)
 
    # Join all the summary sentences into a single string and return it.
    return ' '.join(summary_sentences)

Balance data: Using Resampling to deal with the imbalance dataset. Given 8000 negative and 100 positive data, I use RandomOverSampler to increase the # of positive data to 300. But the dataset is still imbalanced. Since the Resampling method just duplicates the data in minority class, to avoid over-fitting, I choose not to resampling too much data. But I think resampling is not enough to deal with this unbalanced dataset.
sampler = RandomOverSampler(sampling_strategy={'true': 300})

Temperature: Using 0.001 since low temperature is suitable for this binary classification task.

Below is the description of the fine-tuning model(third model).
Our predictions are generated from this model.
Prompt: Based on the following medical notes, please predict whether the patient described is likely to have Acute Respiratory Distress Syndrome (ARDS). Your prediction should be either 'true' if ARDS is likely, or 'false' if it is not likely."


Truncation & Balance data: Using Chris Liang’s Method 2. 
Temperature: 0.01
Prediction method:


Team Reflection:
The prediction results we obtained are not as desired. Fine-tuning an LLM is a complicated task; there are many things that need to be considered.
