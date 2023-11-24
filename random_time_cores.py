import lz4.frame
import gzip
import zstandard as zstd
import os
import pandas as pd
import time
from multiprocessing import Pool

def compress_chunk(args):
    chunk, file_name, compression_algorithm, cost_save_path, i = args
    if compression_algorithm == 'lz4':
        compressed_data = lz4.frame.compress(chunk.to_csv(index=False).encode())
    elif compression_algorithm == 'gzip':
        compressed_data = gzip.compress(chunk.to_csv(index=False).encode())
    elif compression_algorithm == 'zstd':
        cctx = zstd.ZstdCompressor(level=3)
        compressed_data = cctx.compress(chunk.to_csv(index=False).encode())
    else:
        raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")

def CostModel (
        block_size: int,    # the size of total file, MB
        time_limit: int,    # time limit of the data migration, seconds
        num_cores: int,     # number of cpu cores in the source system of data migration
        net_speed: float,   # the network bandwidth from the source system to the target system in data migration, MB/s
        process_speed: float,   # including preprocess and compression, MB/s
        compression_ratio: float,   # final compression ratio
        price_cpu: float = 0.048,   # the price of using one CPU core per hour
        price_net: float = 0.05     # the price of transforming data per gigabyte
):
    compression_ratio = max (compression_ratio, block_size / (time_limit * net_speed), 1)    # compression ratio limitation
    if (process_speed < (block_size / (time_limit * num_cores))):
        print ('The processing speed should be faster due to the time limit!')
        return

    # cost with preprocessing
    cost_with_pre = block_size * price_cpu/ process_speed + block_size * price_net / compression_ratio

    # print the costs
    print(f'The cost with preprocessing is {cost_with_pre:.3f} dollars \nThe cost without proprcessing is {cost_without_pre:.3f} dollars')

    return cost_with_pre

def classify_and_compress(
        file_path,
        chunk_size,
        model,
        max_size,
        compression_algorithm,
        model_name,
        file_name,
):
    worker_data = {worker_id: pd.DataFrame() for worker_id in range(10)}
    compress_total_time = 0

    for i, chunk in enumerate(pd.read_csv(f'{file_path}/{file_name}.csv', chunksize=chunk_size, delimiter='|')):
        processed_data = preprocess_data(chunk)
        predictions, classification_time = classify_data(processed_data, model)
        chunk_with_labels = add_labels_to_chunk(chunk, predictions)

        # add data to workers
        for worker_id, group in chunk_with_labels.groupby('label'):
            worker_data[worker_id] = pd.concat([worker_data[worker_id], group])

            # check the data size, compress and output them once reach max_size
            if len(worker_data[worker_id]) >= max_size:
                compress_time, _ = compress_and_save_worker_data(
                    worker_data=worker_data, 
                    worker_id=worker_id, 
                    compression_algorithm=compression_algorithm, 
                    model_name=model_name,
                    file_name=file_name,
                    train_percent=train_percent
                )
                compress_total_time += compress_time
                # clear the worker after data compressed
                worker_data[worker_id] = pd.DataFrame()

    # after the main loop, compress and output the remaining data
    for worker_id, data in worker_data.items():
        if not data.empty:
            compress_time, save_folder = compress_and_save_worker_data(
                worker_data=worker_data, 
                worker_id=worker_id, 
                compression_algorithm=compression_algorithm, 
                model_name=model_name,
                file_name=file_name,
                train_percent=train_percent
            )
            compress_total_time += compress_time
    # save_folder = os.path.join(compress_save_path, f'{model_name}_{file_name}')
    compressed_size = get_folder_size(save_folder)
    print('save folder: ', save_folder)
    print('compressed_size: ', compressed_size)
    return classification_time, compress_total_time, compressed_size

if __name__ == "__main__":
    file_path = 'YorkProject/data/original'
    file_names = ['econbiz', 'partsupp', 'orders']
    compression_algorithms = ['lz4', 'gzip', 'zstd']
    cost_save_path = 'YorkProject/cost_data'

    for file_name in file_names:
        for compression_algorithm in compression_algorithms:
            df = pd.read_csv(f'{file_path}/{file_name}.csv', sep='|')

            # random split
            total_rows = len(df)
            chunk_size = total_rows // 10
            df_shuffled = df.sample(frac=1, random_state=42)
            chunks = [df_shuffled[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]

            

            num_cores = [1, 2, 4, 8] 
            for num_core in num_cores:
                random_total_time = 0
                pool = Pool(processes=num_core)

                args_list = [(chunk, file_name, compression_algorithm, cost_save_path, i) for i, chunk in enumerate(chunks)]

                start_time = time.time()
                pool.map(compress_chunk, args_list)

                end_time = time.time()
                compress_time = end_time - start_time
                random_total_time += compress_time
                pool.close()    
                pool.join()

                print(f"random time for {file_name} for {compression_algorithm} compression with {num_core} cores: {random_total_time:.2f} seconds")
