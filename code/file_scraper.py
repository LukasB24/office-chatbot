def chunk_csv_file(file_path, chunk_size):
    """
    Chunk a csv file into smaller files of size chunk_size
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, chunk in enumerate(chunked(reader, chunk_size)):
            with open(f'{file_path}_{i}.csv', 'w') as chunk_file:
                writer = csv.writer(chunk_file)
                writer.writerow(header)
                writer.writerows(chunk)