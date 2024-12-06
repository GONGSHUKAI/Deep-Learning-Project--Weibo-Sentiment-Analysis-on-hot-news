import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TransformerModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_size=768, 
                 num_heads=8, 
                 num_layers=6, 
                 num_classes=2, 
                 ff_size=512, 
                 dropout=0.1, 
                 use_pretrained_bert=False, 
                 freeze_bert=True,
                 pretrained_model_name='bert-base-chinese'):
        super(TransformerModel, self).__init__()

        self.use_pretrained_bert = use_pretrained_bert
        
        if self.use_pretrained_bert:
            self.bert = BertModel.from_pretrained(pretrained_model_name)
            # freeze pretrained BERT parameters, use it only for embedding generation
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.embedding_size = self.bert.config.hidden_size  # output dimension of BERT
        else:
            # use simple nn.Embedding
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.embedding_size = embed_size
        
        # Transformer encoder layers
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True
        )
        # Transformer encoder which consists of multiple layers
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers,
            num_layers=num_layers
        )
        
        # FC for binary classification
        self.fc = nn.Linear(self.embedding_size, num_classes)
        self.output_dropout = nn.Dropout(p=0.3)

    def forward(self, x, attention_mask=None, method='mean'):
        if self.use_pretrained_bert:
            outputs = self.bert(input_ids=x, attention_mask=attention_mask)
            x = outputs.last_hidden_state  # shape: (batch_size, seq_len, embed_size)
        else:
            x = self.embedding(x)  # shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        
        x = self.transformer_encoder(x)  # shape: (batch_size, seq_len, embed_size)
        x = self.output_dropout(x)
        
        # Pooling：'mean' or 'last'
        if method == 'mean':
            x = x.mean(dim=1) 
        elif method == 'last':
            x = x[:, -1, :]
        
        x = self.fc(x)
        return x

if __name__ == '__main__':
    vocab_size = BertTokenizer.from_pretrained('bert-base-chinese').vocab_size
    model = TransformerModel(vocab_size=vocab_size, use_pretrained_bert=True)
    print(model)