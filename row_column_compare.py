import pandas as pd
import joblib
import time
import lz4.frame
import gzip
import zstandard as zstd
import os
import matplotlib.pyplot as plt


'''paths'''
model_path = '/Users/fightfei/PycharmProjects/YorkProject/models'
file_path = '/Users/fightfei/PycharmProjects/YorkProject/data/original'
compress_save_path = '/Users/fightfei/PycharmProjects/YorkProject/data/compressed_data'
fig_save_path = '/Users/fightfei/PycharmProjects/YorkProject/figures'
results_path = '/Users/fightfei/PycharmProjects/YorkProject/results'

'''paras'''
chunk_size = 10000
max_size = 10000


'''functions'''
# save results
def save_results(
        file_name: str,
        content: str,
        results_path=results_path
):
    with open(results_path + file_name, 'a') as f:
        f.writelines(content + '\n')

# calculate throughput
def calculate_throughput(
        data_size: float,
        classification_time: float,
        compression_time: float,
        compression_ratio: float,
        network_speed: float,
):
    total_time = classification_time + compression_time
    class_throughput = data_size / total_time
    netwrok_throughput = network_speed * compression_ratio
    throughput = min(class_throughput, netwrok_throughput)
    return throughput

def calculate_cost(
        compression_time: float,
        compression_ratio: float,
        original_size: float,
        num_cores: int,
        cost_scale = 'TB', # calculate the cost of handreds of TBs
        p_cpu = 0.048,
        p_net = 0.05,
):

    base_cost = ((p_cpu / 3600) * num_cores) * compression_time + ((original_size) * p_net / compression_ratio)
    cost = base_cost * 1024 * 1024
    return cost

# keep the columns with numbers only
def preprocess_data(df):
    numeric_columns = df.select_dtypes(include='number')
    return numeric_columns

# classify the data with the loaded model
def classify_data(df, model):
    start_time = time.time()
    predictions = model.predict(df)
    end_time = time.time()
    classification_time = end_time - start_time
    return predictions, classification_time


# add labels to the original data
def add_labels_to_chunk(chunk, labels):
    chunk['label'] = labels
    return chunk

# compute the size of the datas
# return MB
def get_folder_size(folder_path):
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)

    return total_size / (1024 * 1024)

# compress the data in the worker with lz4, gzip, or zstandard
# row compress
def row_compress_and_save_worker_data(
        worker_data, 
        worker_id, 
        compression_algorithm, 
        model_name, 
        file_name,
        train_percent,
):
    data = worker_data[worker_id]
    # drop the label column
    data = data.drop(columns=['label'])

    start_time = time.time()
    if compression_algorithm == 'lz4':
        compressed_data = lz4.frame.compress(data.to_csv(index=False).encode())
    elif compression_algorithm == 'gzip':
        compressed_data = gzip.compress(data.to_csv(index=False).encode())
    elif compression_algorithm == 'zstd':
        cctx = zstd.ZstdCompressor(level=3)
        compressed_data = cctx.compress(data.to_csv(index=False).encode())
    else:
        raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")
    end_time = time.time()
    row_compress_time = end_time - start_time

    # check if there is a folder
    save_folder = os.path.join(compress_save_path, f'{model_name}_{file_name}/row_{compression_algorithm}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    output_name = f'worker_{worker_id}_compressed.{compression_algorithm}'
    output_path = os.path.join(save_folder, output_name)

    with open(output_path, 'ab') as f:
        f.write(compressed_data)

    return row_compress_time, save_folder

# convert columns
def column_to_bytes(column):
    return column.to_string(index=False).encode()

def gzip_compress(data):
    return gzip.compress(data)

def zstd_compress(data):
    compressor = zstd.ZstdCompressor(level=3)
    return compressor.compress(data)

# column compression
def col_compress_and_save_worker_data(
        worker_data, 
        worker_id, 
        compression_algorithm, 
        model_name, 
        file_name,
        train_percent,
):
    data = worker_data[worker_id]
    # drop the label column
    data = data.drop(columns=['label'])

    start_time = time.time()
    if compression_algorithm == 'lz4':
        compressed_data = {col: lz4.frame.compress(column_to_bytes(data[col])) for col in data.columns}
    elif compression_algorithm == 'gzip':
        compressed_data = {col: gzip_compress(column_to_bytes(data[col])) for col in data.columns}
    elif compression_algorithm == 'zstd':
        compressed_data = {col: zstd_compress(column_to_bytes(data[col])) for col in data.columns}
    else:
        raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")
    end_time = time.time()
    col_compress_time = end_time - start_time

    # check if there is a folder
    save_folder = os.path.join(compress_save_path, f'{model_name}_{file_name}/col_{compression_algorithm}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    output_name = f'worker_{worker_id}_compressed.{compression_algorithm}'
    output_path = os.path.join(save_folder, output_name)

    for _, compressed_df in compressed_data.items():
        with open(output_path, 'ab') as f:
            f.write(compressed_df)

    return col_compress_time, save_folder

# classify and compress the data 
def classify_and_row_compress(
        file_path,
        chunk_size,
        model,
        max_size,
        compression_algorithm,
        model_name,
        file_name,
        train_percent,
):
    worker_data = {worker_id: pd.DataFrame() for worker_id in range(10)}

    row_compress_total_time = 0
    classification_total_time = 0

    for i, chunk in enumerate(pd.read_csv(f'{file_path}/{file_name}.csv', chunksize=chunk_size, delimiter='|')):
        processed_data = preprocess_data(chunk)
        predictions, classification_time = classify_data(processed_data, model)
        chunk_with_labels = add_labels_to_chunk(chunk, predictions)

        classification_total_time += classification_time

        # add data to workers
        for worker_id, group in chunk_with_labels.groupby('label'):
            worker_data[worker_id] = pd.concat([worker_data[worker_id], group])

            # check the data size, compress and output them once reach max_size
            if len(worker_data[worker_id]) >= max_size:
                row_compress_time, _ = row_compress_and_save_worker_data(
                    worker_data=worker_data, 
                    worker_id=worker_id, 
                    compression_algorithm=compression_algorithm, 
                    model_name=model_name,
                    file_name=file_name,
                    train_percent=train_percent
                )
                row_compress_total_time += row_compress_time
                # clear the worker after data compressed
                worker_data[worker_id] = pd.DataFrame()

    # after the main loop, compress and output the remaining data
    for worker_id, data in worker_data.items():
        if not data.empty:
            row_compress_time, save_folder = row_compress_and_save_worker_data(
                worker_data=worker_data, 
                worker_id=worker_id, 
                compression_algorithm=compression_algorithm, 
                model_name=model_name,
                file_name=file_name,
                train_percent=train_percent
            )
            row_compress_total_time += row_compress_time
    # save_folder = os.path.join(compress_save_path, f'{model_name}_{file_name}')
    row_compressed_size = get_folder_size(save_folder)
    print('save folder: ', save_folder)
    print('compressed_size: ', row_compressed_size)
    return classification_total_time, row_compress_total_time, row_compressed_size

# classify and compress the data by column
def classify_and_col_compress(
        file_path,
        chunk_size,
        model,
        max_size,
        compression_algorithm,
        model_name,
        file_name,
        train_percent,
):
    worker_data = {worker_id: pd.DataFrame() for worker_id in range(10)}

    col_compress_total_time = 0
    classification_total_time = 0

    for i, chunk in enumerate(pd.read_csv(f'{file_path}/{file_name}.csv', chunksize=chunk_size, delimiter='|')):
        processed_data = preprocess_data(chunk)
        predictions, classification_time = classify_data(processed_data, model)
        chunk_with_labels = add_labels_to_chunk(chunk, predictions)

        classification_total_time += classification_time

        # add data to workers
        for worker_id, group in chunk_with_labels.groupby('label'):
            worker_data[worker_id] = pd.concat([worker_data[worker_id], group])

            # check the data size, compress and output them once reach max_size
            if len(worker_data[worker_id]) >= max_size:
                col_compress_time, _ = col_compress_and_save_worker_data(
                    worker_data=worker_data, 
                    worker_id=worker_id, 
                    compression_algorithm=compression_algorithm, 
                    model_name=model_name,
                    file_name=file_name,
                    train_percent=train_percent
                )
                col_compress_total_time += col_compress_time
                # clear the worker after data compressed
                worker_data[worker_id] = pd.DataFrame()

    # after the main loop, compress and output the remaining data
    for worker_id, data in worker_data.items():
        if not data.empty:
            col_compress_time, save_folder = col_compress_and_save_worker_data(
                worker_data=worker_data, 
                worker_id=worker_id, 
                compression_algorithm=compression_algorithm, 
                model_name=model_name,
                file_name=file_name,
                train_percent=train_percent
            )
            col_compress_total_time += col_compress_time
    # save_folder = os.path.join(compress_save_path, f'{model_name}_{file_name}')
    col_compressed_size = get_folder_size(save_folder)
    print('save folder: ', save_folder)
    print('compressed_size: ', col_compressed_size)
    return classification_total_time, col_compress_total_time, col_compressed_size

# random split and compress by row
def random_split_and_row_compress(
        file_path,
        compression_algorithm,
        file_name,
        train_percent,
):
    df = pd.read_csv(f'{file_path}/{file_name}.csv', sep='|')

    # random split
    total_rows = len(df)
    chunk_size = total_rows // 10
    df_shuffled = df.sample(frac=1, random_state=42)
    chunks = [df_shuffled[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]

    row_random_total_time = 0
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
        row_compress_time = end_time - start_time
        row_random_total_time += row_compress_time
        
        # check if there is a folder
        save_folder = os.path.join(compress_save_path, f'random_{file_name}/row_{compression_algorithm}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        output_name = f'{file_name}_{i}.{compression_algorithm}'
        output_path = os.path.join(save_folder, output_name)

        with open(output_path, 'ab') as f:
            f.write(compressed_data)

    folder_size = get_folder_size(save_folder)
    print('random folder size: ', folder_size)
    return folder_size, row_random_total_time

# random split and compress by column
def random_split_and_col_compress(
        file_path,
        compression_algorithm,
        file_name,
        train_percent,
):
    df = pd.read_csv(f'{file_path}/{file_name}.csv', sep='|')

    # random split
    total_rows = len(df)
    chunk_size = total_rows // 10
    df_shuffled = df.sample(frac=1, random_state=42)
    chunks = [df_shuffled[i:i+chunk_size] for i in range(0, total_rows, chunk_size)]

    col_random_total_time = 0
    for i, chunk in enumerate(chunks):
        start_time = time.time()
        if compression_algorithm == 'lz4':
            compressed_data = {col: lz4.frame.compress(column_to_bytes(chunk[col])) for col in chunk.columns}
        elif compression_algorithm == 'gzip':
            compressed_data = {col: gzip_compress(column_to_bytes(chunk[col])) for col in chunk.columns}
        elif compression_algorithm == 'zstd':
            compressed_data = {col: zstd_compress(column_to_bytes(chunk[col])) for col in chunk.columns}
        else:
            raise ValueError(f"Unsupported compression algorithm: {compression_algorithm}")

        end_time = time.time()
        col_compress_time = end_time - start_time
        col_random_total_time += col_compress_time
        
        # check if there is a folder
        save_folder = os.path.join(compress_save_path, f'random_{file_name}/col_{compression_algorithm}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        output_name = f'{file_name}_{i}.{compression_algorithm}'
        output_path = os.path.join(save_folder, output_name)

        for _, compressed_df in compressed_data.items():
            with open(output_path, 'ab') as f:
                f.write(compressed_df)

    folder_size = get_folder_size(save_folder)
    print('random folder size: ', folder_size)
    return folder_size, col_random_total_time


'''main function'''
def main():
    model_names = ['DecisionTree','RandomForest','AdaBoost','KNN','LogisticRegression','MLP']
    compression_algorithms = ['gzip', 'lz4', 'zstd']
    # compression_algorithms = ['zstd']
    file_names = ['econbiz', 'partsupp', 'orders']
    # file_names = ['orders']

    # train_percents = [20, 40, 60, 80, 90]
    train_percents = [20]

    for train_percent in train_percents:
        for file_name in file_names:
            original_data_sizes = {
                    # 'cosumer complain': 755.8,
                    'econbiz': 151.3,
                    'orders': 170.5,
                    'partsupp': 118.2
                }
            if file_name in original_data_sizes:
                original_data_size = original_data_sizes[file_name]

                # set the result path
            result_save_folder = os.path.join(results_path, f'{train_percent}%_train')
            if not os.path.exists(result_save_folder):
                os.makedirs(result_save_folder)

            # set network speed list
            network_speeds = [2, 5, 10, 15, 20, 25]

            for compression_algorithm in compression_algorithms:

                row_throughput_dict = {} # store the throughputs
                col_throughput_dict = {} # store the throughputs
                row_cost_dict = {} # store the cost
                col_cost_dict = {} # store the cost

                # calculate random throughput
                row_random_throughputs = []
                col_random_throughputs = []
                # calculate random cost
                row_random_costs = []
                col_random_costs = []

                row_random_compressed_size, row_random_compress_time = random_split_and_row_compress(
                    file_path,
                    compression_algorithm,
                    file_name,
                    train_percent,
                )
                col_random_compressed_size, col_random_compress_time = random_split_and_col_compress(
                    file_path,
                    compression_algorithm,
                    file_name,
                    train_percent,
                )

                row_random_cr = original_data_size / row_random_compressed_size
                col_random_cr = original_data_size / col_random_compressed_size

                for network_speed in network_speeds:
                    # row
                    throughput = calculate_throughput(
                        data_size=original_data_size,
                        classification_time=0,
                        compression_time=row_random_compress_time,
                        compression_ratio=row_random_cr,
                        network_speed=network_speed,
                    )
                    row_random_throughputs.append(round(throughput, 2))

                    cost = calculate_cost(
                        compression_time=row_random_compress_time,
                        compression_ratio=row_random_cr,
                        original_size=original_data_sizes[file_name],
                        num_cores=8,
                    )
                    row_random_costs.append(round(cost, 2))

                    # column
                    throughput = calculate_throughput(
                        data_size=original_data_size,
                        classification_time=0,
                        compression_time=col_random_compress_time,
                        compression_ratio=col_random_cr,
                        network_speed=network_speed,
                    )
                    col_random_throughputs.append(round(throughput, 2))

                    cost = calculate_cost(
                        compression_time=col_random_compress_time,
                        compression_ratio=col_random_cr,
                        original_size=original_data_sizes[file_name],
                        num_cores=8,
                    )
                    col_random_costs.append(round(cost, 2))

                print('col_random_cr: ', col_random_cr)
                print('col_random_throughputs: ', col_random_throughputs)
                print('col_random_costs: ', col_random_costs)
                print('row_random_cr: ', row_random_cr)
                print('row_random_throughputs: ', row_random_throughputs)
                print('row_random_costs: ', row_random_costs)

                # save the results
                with open(f'{result_save_folder}/{file_name}_{compression_algorithm}.txt', 'a') as fout:
                    fout.writelines(f'original data size: \n')
                    for name, size in original_data_sizes.items():
                        fout.writelines(f'{name}: {size:} MB\n')
                    
                    fout.writelines(f'\n')
                    fout.writelines(f'row random split compression time: {row_random_compress_time:.2f}\n')
                    fout.writelines(f'row random split compressed size: {row_random_compressed_size:.2f}\n')
                    fout.writelines(f'row random split compression ratio: {row_random_cr:.2f}\n')
                    fout.writelines(f'row random split throughput list: {row_random_throughputs}\n')
                    fout.writelines(f'row random split cost list: {row_random_costs}\n')
                    fout.writelines(f'\n')
                    fout.writelines(f'col random split compression time: {col_random_compress_time:.2f}\n')
                    fout.writelines(f'col random split compressed size: {col_random_compressed_size:.2f}\n')
                    fout.writelines(f'col random split compression ratio: {col_random_cr:.2f}\n')
                    fout.writelines(f'col random split throughput list: {col_random_throughputs}\n')
                    fout.writelines(f'col random split cost list: {col_random_costs}\n')
                    fout.writelines(f'\n')
                    fout.writelines(f'network speed list: {network_speeds}\n')


                row_throughput_dict['random'] = row_random_throughputs
                col_throughput_dict['random'] = col_random_throughputs
                col_cost_dict['random'] = col_random_costs
                row_cost_dict['random'] = row_random_costs

                # classify using the classifiers
                #####
                for model_name in model_names:

                    model = joblib.load(f'{model_path}/{train_percent}%_train/{model_name}_{file_name}.joblib')

                    # simulate the multi-worker situation
                    row_classification_time, row_compress_total_time, row_compressed_size = classify_and_row_compress(
                        file_path=file_path,
                        chunk_size=chunk_size,
                        model=model,
                        max_size=max_size,
                        compression_algorithm=compression_algorithm,
                        model_name=model_name,
                        file_name=file_name,
                        train_percent=train_percent,
                    )
                    
                    row_compression_ratio = original_data_size / row_compressed_size
                    print('row compression_ratio: ', row_compression_ratio)

                    row_classifier_throughputs = []
                    row_classifier_costs = []
                    for network_speed in network_speeds:
                        # compute throughputs
                        throughput = calculate_throughput(
                            data_size=original_data_size,
                            classification_time=row_classification_time,
                            compression_time=row_compress_total_time,
                            compression_ratio=row_compression_ratio,
                            network_speed=network_speed,
                        )
                        row_classifier_throughputs.append(round(throughput, 2))

                        # compute cost
                        cost = calculate_cost(
                            compression_time=row_classification_time+row_compress_total_time,
                            compression_ratio=row_compression_ratio,
                            original_size=original_data_sizes[file_name],
                            num_cores=8,
                        )
                        row_classifier_costs.append(round(cost, 2))


                    row_throughput_dict[f'{model_name}'] = row_classifier_throughputs
                    row_cost_dict[f'{model_name}'] = row_classifier_costs
                    print('row classifier_throughputs: ', row_classifier_throughputs)
                    print('row classifier_cost: ', row_classifier_costs)

                    # column compression
                    col_classification_time, col_compress_total_time, col_compressed_size = classify_and_col_compress(
                        file_path=file_path,
                        chunk_size=chunk_size,
                        model=model,
                        max_size=max_size,
                        compression_algorithm=compression_algorithm,
                        model_name=model_name,
                        file_name=file_name,
                        train_percent=train_percent,
                    )
                    
                    col_compression_ratio = original_data_size / col_compressed_size
                    print('col compression_ratio: ', col_compression_ratio)

                    col_classifier_throughputs = []
                    col_classifier_costs = []
                    for network_speed in network_speeds:
                        # compute throughputs
                        throughput = calculate_throughput(
                            data_size=original_data_size,
                            classification_time=col_classification_time,
                            compression_time=col_compress_total_time,
                            compression_ratio=col_compression_ratio,
                            network_speed=network_speed,
                        )
                        col_classifier_throughputs.append(round(throughput, 2))

                        # compute cost
                        cost = calculate_cost(
                            compression_time=col_classification_time+col_compress_total_time,
                            compression_ratio=col_compression_ratio,
                            original_size=original_data_sizes[file_name],
                            num_cores=8,
                        )
                        col_classifier_costs.append(round(cost, 2))


                    col_throughput_dict[f'{model_name}'] = col_classifier_throughputs
                    col_cost_dict[f'{model_name}'] = col_classifier_costs
                    print('col classifier_throughputs: ', col_classifier_throughputs)
                    print('col classifier_cost: ', col_classifier_costs)

                    # save the outputs
                    with open(f'{result_save_folder}/{file_name}_{compression_algorithm}.txt', 'a') as fout:
                        fout.writelines(f'\n')
                        fout.writelines(f'classifier: {model_name}\n')
                        fout.writelines(f'row classification time: {row_classification_time:.2f}\n')
                        fout.writelines(f'row compression time: {row_compress_total_time:.2f}\n')
                        fout.writelines(f'row compressed size: {row_compressed_size:.2f}\n')
                        fout.writelines(f'row compression ratio: {row_compression_ratio:.2f}\n')
                        fout.writelines(f'row throughput list: {row_classifier_throughputs}\n')
                        fout.writelines(f'row cost list: {row_classifier_costs}\n')
                        
                        fout.writelines(f'col classification time: {col_classification_time:.2f}\n')
                        fout.writelines(f'col compression time: {col_compress_total_time:.2f}\n')
                        fout.writelines(f'col compressed size: {col_compressed_size:.2f}\n')
                        fout.writelines(f'col compression ratio: {col_compression_ratio:.2f}\n')
                        fout.writelines(f'col throughput list: {col_classifier_throughputs}\n')
                        fout.writelines(f'col cost list: {col_classifier_costs}\n')
                #####
                print(row_throughput_dict)
                print(col_throughput_dict)

                # show the throughput figure and save
                plt.figure(figsize=(10, 6))

                for key, values in row_throughput_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Row Compression Throughput')
                plt.title(f'{file_name}_row_{compression_algorithm}_{train_percent}%_Throughput')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_row_{compression_algorithm}_throughput.png')
                # plt.show()

                # column figure
                plt.figure(figsize=(10, 6))

                for key, values in col_throughput_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Column Compression Throughput')
                plt.title(f'{file_name}_col_{compression_algorithm}_{train_percent}%_Throughput')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_col_{compression_algorithm}_throughput.png')


                # show the cost figure and save
                plt.figure(figsize=(10, 6))

                for key, values in row_cost_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Row Compression Cost(TBs)')
                plt.title(f'{file_name}_col_{compression_algorithm}_{train_percent}%_Cost(TBs)')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_row_{compression_algorithm}_cost.png')

                # column compression
                plt.figure(figsize=(10, 6))

                for key, values in col_cost_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Col Compression Cost(TBs)')
                plt.title(f'{file_name}_col_{compression_algorithm}_{train_percent}%_Cost(TBs)')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_col_{compression_algorithm}_cost.png')

        

if __name__ == "__main__":
    main()