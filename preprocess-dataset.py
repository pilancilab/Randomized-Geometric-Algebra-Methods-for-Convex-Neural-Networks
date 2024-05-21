import os
import re
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertModel
import argparse
import openai
from openai import OpenAI

# paste your openAI API key and ogranization code below.
API_key = ''
ORGANIZATION = ''

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument("--data_name", type=str, default='IMDB', choices=['IMDB','AmazonPolarity','cvx-forum','glue-cola','glue-qqp'])
    parser.add_argument("--data_path", type=str, default = './data')
    parser.add_argument("--export_num", type=str, default = '10',choices=['10','full','30K','21','50K'])
    parser.add_argument("--embedding", type=str, default = 'Bert', choices=['Bert','OpenAI'])
    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    data_name = args.data_name

    if data_name == 'IMDB':
        # Load data and set labels
        Data_dir = "{}/IMDB_data.csv".format(args.data_path)
        df = pd.read_csv(Data_dir)
        df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    elif data_name == 'AmazonPolarity':
        Data_dir = "{}/Amazon_polarity_train.csv".format(args.data_path)
        df = pd.read_csv(Data_dir)
        df['review'] = df['content']
    elif data_name == 'cvx-forum':
        Data_dir = "{}/cvx-forum-QA-new.csv".format(args.data_path)
        df = pd.read_csv(Data_dir)
        df['review'] = df['text']
    elif data_name == 'glue-cola':
        Data_dir = "{}/glue_cola_raw.csv".format(args.data_path)
        df = pd.read_csv(Data_dir)
        df['review'] = df['sentence']
    elif data_name == 'glue-qqp':
        Data_dir = "{}/glue_qqp_raw.csv".format(args.data_path)
        df = pd.read_csv(Data_dir)
        df['review'] = 'Question 1: ' + df['question1']+ '\n Question 2: ' +df['question2']

    # truncate to first 1000 samples
    if args.export_num=='10':
        df = df[:10]
    if args.export_num=='21':
        df = df[:21]
    elif args.export_num=='30K':
        df = df.sample(30000,random_state=0)
    elif args.export_num=='50K':
        df = df.sample(50000,random_state=0)

    if args.embedding=='Bert':
        from transformers import BertTokenizer

        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Create a function to tokenize a set of texts
        def preprocessing_for_bert(data):
            """Perform required preprocessing steps for pretrained BERT.
            @param    data (np.array): Array of texts to be processed.
            @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
            @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                          tokens should be attended to by the model.
            """
            # Create empty lists to store outputs
            input_ids = []
            attention_masks = []



            # For every sentence...
            for sent in tqdm(data):
                try:
                    text = text_preprocessing(sent)
                except:
                    text = sent
                    print(sent)
                # `encode_plus` will:
                #    (1) Tokenize the sentence
                #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
                #    (3) Truncate sentence to max length
                #    (4) Map tokens to their IDs
                #    (5) Create attention mask
                #    (6) Return a dictionary of outputs
                encoded_sent = tokenizer.encode_plus(
                    text=text,  # Preprocess sentence
                    add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                    max_length=512,                 # Max length to truncate/pad
                    truncation=True,                # Truncate longer messages
                    pad_to_max_length=True,         # Pad sentence to max length
                    return_attention_mask=True      # Return attention mask
                    )

                # Add the outputs to the lists
                input_ids.append(encoded_sent.get('input_ids'))
                attention_masks.append(encoded_sent.get('attention_mask'))

            # Convert lists to tensors
            input_ids = torch.tensor(input_ids)
            attention_masks = torch.tensor(attention_masks)

            return input_ids, attention_masks

        X = df.review.values

        # Specify `MAX_LEN`
        MAX_LEN = 512

        # Print sentence 0 and its encoded token ids
        token_ids = list(preprocessing_for_bert(X[0])[0].squeeze().numpy())
        # print('Original: ', X[0])
        # print('Token IDs: ', token_ids)

        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        data_inputs, data_masks = preprocessing_for_bert(X)

        # extract Bert embedding
        y = df.label.values
        Pretrained_Bert = BertModel.from_pretrained('bert-base-uncased')

        # Initialize empty arrays for data embeddings
        data_embeddings = np.zeros((0, 768))

        batch_size = 20
        # Extract embeddings for data
        extra_iter = 0

        num_item = data_inputs.shape[0]
        if num_item%batch_size != 0:
            extra_iter = 1
        for i in tqdm(range(num_item // batch_size + extra_iter)):
            outputs = Pretrained_Bert(input_ids=data_inputs[i * batch_size:min((i + 1) * batch_size,num_item)], attention_mask=data_masks[i * batch_size:min((i + 1) * batch_size,num_item)])
            cls_embeddings = outputs[0][:, 0, :].squeeze().detach().numpy().reshape([-1,768])
            data_embeddings = np.concatenate((data_embeddings, cls_embeddings))
        # Create a DataFrame for data embeddings and labels
        data_df = pd.DataFrame(data_embeddings)
        data_df['Label'] = y

        csv_path='{}/{}-{}-{}-Embeddings.csv'.format(args.data_path,data_name, args.embedding,args.export_num)
        # Save training embeddings to CSV
        data_df.to_csv(csv_path, index=False)

    if args.embedding == 'OpenAI':
        client = OpenAI(api_key = API_key, organization=ORGANIZATION)
        X = df.review.values
        y = df.label.values

        dfX = pd.DataFrame()
        dfX["Text"] = X
        dfX["Labels"] = y

        def get_embedding(text, model="text-embedding-ada-002"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model).data[0].embedding

        tqdm.pandas()
        dfX["Embedding"] = dfX.Text.progress_apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
        dfZ = pd.DataFrame()
        dfZ["Embedding"] = dfX["Embedding"]
        dfZ["Labels"] = dfX["Labels"]

        dfZ = pd.DataFrame()
        dfZ["Embedding"] = dfX["Embedding"]
        dfZ["Labels"] = dfX["Labels"]

        csv_path = '{}/{}-{}-{}-Raw-Embeddings.csv'.format(args.data_path,data_name, args.embedding,args.export_num)
        dfZ.to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)

        # Split the 'test' column by commas and expand into separate columns
        split_df = df['Embedding'].str.split(',', expand=True)

        # Rename the new columns with meaningful names
        split_df.columns = [f'{i}' for i in range(len(split_df.columns))]

        df.drop(columns=['Embedding'], inplace=True)

        # Concatenate the split DataFrame with the original DataFrame
        df = pd.concat([split_df, df], axis=1)

        # print(df.head(5))

        df_new = df.copy()

        columns_to_clean = ['0', '1535']
        # Remove '[' and ']' characters from string columns
        df_new[columns_to_clean] = df_new[columns_to_clean].apply(lambda x: x.replace(r'\[|\]', '', regex=True))
        csv_path = '{}/{}-{}-{}-Embeddings.csv'.format(args.data_path,data_name, args.embedding,args.export_num)
        df_new.to_csv(csv_path, index=False)



if __name__ == '__main__':
    main()