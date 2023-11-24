import pandas as pd
import joblib
import time
import lz4.frame
import gzip
import zstandard as zstd
import os
import matplotlib.pyplot as plt


'''paths'''
model_path = 'YorkProject/models'
file_path = 'YorkProject/data/original'
compress_save_path = 'YorkProject/data/compressed_data'
fig_save_path = 'YorkProject/figures'
results_path = 'YorkProject/results/'

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
def compress_and_save_worker_data(
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
    compress_time = end_time - start_time

    # check if there is a folder
    save_folder = os.path.join(compress_save_path, f'{train_percent}%_train/{model_name}_{file_name}/{compression_algorithm}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    output_name = f'worker_{worker_id}_compressed.{compression_algorithm}'
    output_path = os.path.join(save_folder, output_name)

    with open(output_path, 'ab') as f:
        f.write(compressed_data)

    return compress_time, save_folder

# classify and compress the data 
def classify_and_compress(
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
    compress_total_time = 0

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
    return classification_total_time, compress_total_time, compressed_size

# random split 
def random_split_and_compress(
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
        save_folder = os.path.join(compress_save_path, f'{train_percent}%_train/random_{file_name}/{compression_algorithm}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        output_name = f'{file_name}_{i}.{compression_algorithm}'
        output_path = os.path.join(save_folder, output_name)

        with open(output_path, 'ab') as f:
            f.write(compressed_data)

    folder_size = get_folder_size(save_folder)
    print('random folder size: ', folder_size)
    return folder_size, random_total_time


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

                throughput_dict = {} # store the throughputs
                cost_dict = {} # store the cost

                # calculate random throughput
                random_throughputs = []
                # calculate random cost
                random_costs = []
                random_compressed_size, random_compress_time = random_split_and_compress(
                    file_path,
                    compression_algorithm,
                    file_name,
                    train_percent,
                )

                random_cr = original_data_size / random_compressed_size
                for network_speed in network_speeds:
                    throughput = calculate_throughput(
                        data_size=original_data_size,
                        classification_time=0,
                        compression_time=random_compress_time,
                        compression_ratio=random_cr,
                        network_speed=network_speed,
                    )
                    random_throughputs.append(round(throughput, 2))

                    cost = calculate_cost(
                        compression_time=random_compress_time,
                        compression_ratio=random_cr,
                        original_size=original_data_sizes[file_name],
                        num_cores=8,
                    )
                    random_costs.append(round(cost, 2))

                print('random_cr: ', random_cr)
                print('random_throughputs: ', random_throughputs)
                print('random_costs: ', random_costs)

                # save the results
                with open(f'{result_save_folder}/{file_name}_{compression_algorithm}.txt', 'a') as fout:
                    fout.writelines(f'original data size: \n')
                    for name, size in original_data_sizes.items():
                        fout.writelines(f'{name}: {size:} MB\n')
                    
                    fout.writelines(f'\n')
                    fout.writelines(f'random split compression time: {random_compress_time:.2f}\n')
                    fout.writelines(f'random split compressed size: {random_compressed_size:.2f}\n')
                    fout.writelines(f'random split compression ratio: {random_cr:.2f}\n')
                    fout.writelines(f'random split throughput list: {random_throughputs}\n')
                    fout.writelines(f'random split cost list: {random_costs}\n')
                    fout.writelines(f'\n')
                    fout.writelines(f'network speed list: {network_speeds}\n')


                throughput_dict['random'] = random_throughputs
                cost_dict['random'] = random_costs

                # classify using the classifiers
                #####
                for model_name in model_names:

                    model = joblib.load(f'{model_path}/{train_percent}%_train/{model_name}_{file_name}.joblib')

                    # simulate the multi-worker situation
                    classification_time, compress_total_time, compressed_size = classify_and_compress(
                        file_path=file_path,
                        chunk_size=chunk_size,
                        model=model,
                        max_size=max_size,
                        compression_algorithm=compression_algorithm,
                        model_name=model_name,
                        file_name=file_name,
                        train_percent=train_percent,
                    )
                    
                    compression_ratio = original_data_size / compressed_size
                    print('compression_ratio: ', compression_ratio)

                    classifier_throughputs = []
                    classifier_costs = []
                    for network_speed in network_speeds:
                        # compute throughputs
                        throughput = calculate_throughput(
                            data_size=original_data_size,
                            classification_time=classification_time,
                            compression_time=compress_total_time,
                            compression_ratio=compression_ratio,
                            network_speed=network_speed,
                        )
                        classifier_throughputs.append(round(throughput, 2))

                        # compute cost
                        cost = calculate_cost(
                            compression_time=classification_time+compress_total_time,
                            compression_ratio=compression_ratio,
                            original_size=original_data_sizes[file_name],
                            num_cores=8,
                        )
                        classifier_costs.append(round(cost, 2))


                    throughput_dict[f'{model_name}'] = classifier_throughputs
                    cost_dict[f'{model_name}'] = classifier_costs
                    print('classifier_throughputs: ', classifier_throughputs)
                    print('classifier_cost: ', classifier_costs)

                    # save the outputs
                    with open(f'{result_save_folder}/{file_name}_{compression_algorithm}.txt', 'a') as fout:
                        fout.writelines(f'\n')
                        fout.writelines(f'classifier: {model_name}\n')
                        fout.writelines(f'classification time: {classification_time:.2f}\n')
                        fout.writelines(f'compression time: {compress_total_time:.2f}\n')
                        fout.writelines(f'compressed size: {compressed_size:.2f}\n')
                        fout.writelines(f'compression ratio: {compression_ratio:.2f}\n')
                        fout.writelines(f'throughput list: {classifier_throughputs}\n')
                        fout.writelines(f'cost list: {classifier_costs}\n')
                #####
                print(throughput_dict)

                # show the throughput figure and save
                plt.figure(figsize=(10, 6))

                for key, values in throughput_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Throughput')
                plt.title(f'{file_name} {compression_algorithm} {train_percent}%_train Throughput')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_{compression_algorithm}_throughput.png')
                # plt.show()

                # show the cost figure and save
                plt.figure(figsize=(10, 6))

                for key, values in cost_dict.items():
                    plt.plot(network_speeds, values, label=key)

                plt.xlabel('Network Speed')
                plt.ylabel('Cost(TBs)')
                plt.title(f'{file_name} {compression_algorithm} {train_percent}%_train Cost(TBs)')
                plt.legend()
                plt.grid(True)
                fig_save_folder = os.path.join(fig_save_path, f'{train_percent}%_train')
                if not os.path.exists(fig_save_folder):
                    os.makedirs(fig_save_folder)
                plt.savefig(f'{fig_save_folder}/{file_name}_{compression_algorithm}_cost.png')

        

if __name__ == "__main__":
    main()