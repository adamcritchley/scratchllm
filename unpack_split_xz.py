import mmap
import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = ""
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()


with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)


with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(chr(char) + os.linesep)

def get_random_chunk(split):
    filename = "train_split.txt" if split == "train" else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size*batch_size - 1)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y