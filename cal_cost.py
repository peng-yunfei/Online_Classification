import pandas as pd
import time
import lz4.frame
import zstandard as zstd
import gzip
import os

file_path = 'YorkProject/data/original'
file_names = ['econbiz']
# file_names = ['econbiz', 'partsupp', 'orders']
compression_algorithms = ['lz4']
# compression_algorithms = ['lz4', 'gzip', 'zstd']
cost_save_path = 'YorkProject/cost_data'


def get_folder_size(folder_path):
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)

    return total_size / (1024 * 1024)

for file_name in file_names:
    for compression_algorithm in compression_algorithms:

        df = pd.read_csv(f'{file_path}/{file_name}.csv', sep='|')

        # random split
        total_rows = len(df)
        chunk_size = total_rows // 10
        df_shuffled = df.sample(frac=1, random_state=42)
        chunks = [df_shuffled[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]

        random_total_time = 0
        for i, chunk in enumerate(chunks):
            start_time = time.time()
            if compression_algorithm == 'lz4':
                compressed_data = lz4.frame.compress(chunk.to_csv(index=False).encode())
            elif compression_algorithm == 'gzip':
                compressed_data = gzip.compress(chunk.to_csv(index=False).encode())
            elif compression_algorithm == 'zstd':
                cctx = zstd.ZstdCompressor(level=3)
                compressed_data = cctx.compress(chunk.to_csv(index=False).encode())
            else:
                raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")

            end_time = time.time()
            compress_time = end_time - start_time
            random_total_time += compress_time
            
            # check if there is a folder
        #     save_folder = os.path.join(cost_save_path, f'random_{file_name}/{compression_algorithm}')
        #     if not os.path.exists(save_folder):
        #         os.makedirs(save_folder)

        #     output_name = f'{file_name}_{i}.{compression_algorithm}'
        #     output_path = os.path.join(save_folder, output_name)

        #     with open(output_path, 'ab') as f:
        #         f.write(compressed_data)

        # folder_size = get_folder_size(save_folder)
        # print('random folder size: ', folder_size)
        print(f'total time: {random_total_time:.2f} s')
