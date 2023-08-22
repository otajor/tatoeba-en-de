# %%
import torch
import sentencepiece as spm


torch.manual_seed(42)

max_seq_length = 50


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Load the SentencePiece models
        self.german_sp = spm.SentencePieceProcessor()
        self.german_sp.Load("de.model")

        self.english_sp = spm.SentencePieceProcessor()
        self.english_sp.Load("en.model")

        # Load the sentences
        with open("de.txt", "r", encoding="utf-8") as f:
            self.german_sentences = f.readlines()

        with open("en.txt", "r", encoding="utf-8") as f:
            self.english_sentences = f.readlines()

        assert len(self.german_sentences) == len(
            self.english_sentences
        ), "Mismatch in number of sentences!"

    def __len__(self):
        return len(self.german_sentences)

    def __getitem__(self, index):
        german_sentence = self.german_sentences[index].strip()
        english_sentence = self.english_sentences[index].strip()

        german_tokens = self.german_sp.EncodeAsIds(german_sentence)
        english_tokens = self.english_sp.EncodeAsIds(english_sentence)

        input = [self.german_sp.bos_id()] + german_tokens
        label = german_tokens + [self.german_sp.eos_id()]

        return {
            "en": english_sentence,
            "de": german_sentence,
            "contx": torch.tensor(english_tokens),
            "input": torch.tensor(input),
            "label": torch.tensor(label),
        }

    def collate_fn(self, batch):
        contx_pad = torch.nn.utils.rnn.pad_sequence(
            [item["contx"] for item in batch], batch_first=True, padding_value=0
        )
        input_pad = torch.nn.utils.rnn.pad_sequence(
            [item["input"] for item in batch], batch_first=True, padding_value=0
        )
        label_pad = torch.nn.utils.rnn.pad_sequence(
            [item["label"] for item in batch], batch_first=True, padding_value=0
        )

        print(contx_pad.shape)
        return {
            "eng": [item["en"] for item in batch],
            "de": [item["de"] for item in batch],
            "contx": contx_pad,
            "input": input_pad,
            "label": label_pad,
        }

    def decode_german(self, tensor):
        ids = tensor.tolist()
        return self.german_sp.DecodeIds(ids)

    def decode_english(self, tensor):
        ids = tensor.tolist()
        return self.english_sp.DecodeIds(ids)

    def get_de_vocab_map(self):
        vocab_map = {}
        for id in range(self.german_sp.get_piece_size()):
            token = self.german_sp.id_to_piece(id)
            vocab_map[id] = token
        return vocab_map

    def get_en_vocab_map(self):
        vocab_map = {}
        for id in range(self.english_sp.get_piece_size()):
            token = self.english_sp.id_to_piece(id)
            vocab_map[id] = token
        return vocab_map


ds = TranslationDataset()

total_size = len(ds)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

trn_ds, tst_ds = torch.utils.data.random_split(ds, [train_size, test_size])


trn_dl = torch.utils.data.DataLoader(
    trn_ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn
)
tst_dl = torch.utils.data.DataLoader(
    tst_ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn
)

# %%

# max_length_en = 0
# max_length_de = 0
# num_long_seqs = 0
# for seq in ds:
#     de_seq = seq["german"]
#     en_seq = seq["english"]
#     if len(de_seq) > 50 or len(en_seq) > 50:
#         num_long_seqs += 1
# if len(de_seq) > max_length_de:
#     max_length_de = len(de_seq)
#     longest_de = de_seq
# if len(en_seq) > max_length_en:
#     max_length_en = len(en_seq)
#     longest_en = en_seq

# print(num_long_seqs)
# # %%
# print(max_length_de, "<<<< de")
# print(max_length_en, "<<<< en")
# print(ds.decode_german(longest_de))
# print(ds.decode_english(longest_en))
# print(ds.get_de_vocab_map())
# print(ds.get_en_vocab_map())


# %%
