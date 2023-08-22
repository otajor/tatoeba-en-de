# %%
import csv
import sentencepiece as spm

# %%
german_sentences = []
english_sentences = []
with open("de_en_sentences.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    data = list(reader)

    for row in data:
        if row[1] and row[3] and row[3].find("\n") == -1:
            german_sentences.append(row[1])
            english_sentences.append(row[3])
        else:
            print(row)
            break


print(f"German: {german_sentences[0]}")
print(f"English: {english_sentences[0]}")
print(f"German: {len(german_sentences)}")
print(f"English: {len(english_sentences)}")

# %%

with open("de.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(german_sentences))

with open("en.txt", "w", encoding="utf-8") as f:
    f.writelines("\n".join(english_sentences))
# %%

de_spm_trainer = spm.SentencePieceTrainer.Train(
    input="de.txt",
    model_type="bpe",
    model_prefix="de",
    vocab_size=10000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
)

spm_trainer = spm.SentencePieceTrainer.Train(
    input="en.txt",
    model_type="bpe",
    model_prefix="en",
    vocab_size=10000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
)

# %%
