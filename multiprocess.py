import os
import gzip
import multiprocessing
import lz4.frame
import zstandard as zstd
import time

def compress_file(filename):
    start_time = time.time()  # 记录开始时间

    with open(filename, 'rb') as f_in:
        cctx = zstd.ZstdCompressor()
        with open(f'{filename}.zst', 'wb') as f_out:
            f_out.write(cctx.compress(f_in.read()))

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算时间差
    print(f"Compressed {filename} in {elapsed_time:.4f} seconds using {multiprocessing.current_process().name}")

def main():
    files_to_compress = ['YorkProject/data/original/cosumer complain.csv']  # 要压缩的文件列表

    core_counts = [1, 2, 4, 8]  # 不同的 CPU 核心数量
    for num_cores in core_counts:
        print(f"Using {num_cores} cores:")
        pool = multiprocessing.Pool(processes=num_cores)

        pool.map(compress_file, files_to_compress)

        pool.close()
        pool.join()
        print("\n")

if __name__ == '__main__':
    main()
