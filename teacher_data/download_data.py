import os
import gzip
import pickle
from huggingface_hub import hf_hub_download, list_repo_files

def download_and_filter_chunks(repo_id, local_dir="downloaded_teacher_data", filter_correct_only=True):

    os.makedirs(local_dir, exist_ok=True)
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        chunk_files = [f for f in files if f.startswith("chunk_") and f.endswith(".pkl.gz")]
        metadata_files = [f for f in files if f.startswith("metadata") and f.endswith(".pkl.gz")]
        
        print(f"Found {len(chunk_files)} chunk files and {len(metadata_files)} metadata files")
        
        for filename in chunk_files + metadata_files:
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
    except Exception as e:
        return False
    
    # Create filtered merged file
    filtered_output = os.path.join(local_dir, "teacher_gsm8k_sparse.pkl.gz")
    
    print(f"Creating filtered merged file...")
    total_samples = 0
    correct_samples = 0
    
    with gzip.open(filtered_output, 'wb', compresslevel=9) as fout:
        # Write metadata header
        metadata_path = os.path.join(local_dir, "metadata.pkl.gz")
        if os.path.exists(metadata_path):
            with gzip.open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                pickle.dump({'metadata': metadata}, fout, protocol=5)
        
        # Process chunks
        chunk_files_local = sorted([f for f in os.listdir(local_dir) if f.startswith("chunk_") and f.endswith(".pkl.gz")])
        
        for chunk_file in chunk_files_local:
            chunk_path = os.path.join(local_dir, chunk_file)
            print(f"Processing {chunk_file}...")
            
            with gzip.open(chunk_path, "rb") as fin:
                try:
                    # Read chunk header
                    chunk_header = pickle.load(fin)
                    samples_in_chunk = chunk_header['num_samples']
                    
                    # Process each sample
                    for i in range(samples_in_chunk):
                        sample = pickle.load(fin)
                        total_samples += 1
                        
                        # Filter for correct answers only if requested
                        if filter_correct_only:
                            if sample.get('answer_correct') == 'yes':
                                pickle.dump(sample, fout, protocol=5)
                                correct_samples += 1
                        else:
                            pickle.dump(sample, fout, protocol=5)
                            correct_samples += 1
                        
                        if total_samples % 100 == 0:
                            print(f"  Processed {total_samples} samples, kept {correct_samples}")
                            
                except Exception as e:
                    print(f"Error processing {chunk_file}: {e}")
                    continue
        
        # Write end marker
        pickle.dump({'_end': True, 'num_samples': correct_samples}, fout, protocol=5)
    
    print(f"\nFiltering complete:")
    print(f"Total samples processed: {total_samples}")
    print(f"Correct samples kept: {correct_samples}")
    print(f"Filtered file saved to: {filtered_output}")
    
    # Clean up individual chunk files to save space
    print("Cleaning up individual chunk files...")
    for chunk_file in chunk_files_local:
        os.remove(os.path.join(local_dir, chunk_file))
    
    return filtered_output

def iter_merged_teacher_data(merged_path: str):
    """
    Lazily iterate samples from the merged stream (same as your training script)
    """
    with gzip.open(merged_path, "rb") as f:
        header = pickle.load(f)  # {'metadata': ...}
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict) and obj.get('_end'):
                break
            yield obj

if __name__ == "__main__":
    repo_id = "lizardp1/gsm8k_early_exit"
    
    filtered_file = download_and_filter_chunks(
        repo_id=repo_id,
        local_dir="downloaded_teacher_data", 
        filter_correct_only=True
    )
    
    if filtered_file:
        print(f"\nTesting the filtered data...")
        count = 0
        for sample in iter_merged_teacher_data(filtered_file):
            if count < 3:
                print(f"Sample {count + 1}:")
                print(f"  Question: {sample['full_user_prompt'][0]}")
                print(f"  Answer correct: {sample['answer_correct']}")
                print(f"  Difficulty: {sample.get('difficulty_category', 'Unknown')}")
            count += 1
            if count >= 100:  # Test first 100
                break
        print(f"Successfully loaded {count} filtered samples")
        
        # Update your training script to use this path:
        print(f"\nUpdate your training script:")
        print(f'teacher_data_path = "{filtered_file}"')