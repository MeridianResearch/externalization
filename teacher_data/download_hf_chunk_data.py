import os
import gzip
import pickle
from huggingface_hub import hf_hub_download, list_repo_files, upload_file
from datasets import load_dataset
import tempfile

def download_chunks_from_hf(repo_id, local_dir="downloaded_teacher_data"):

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
        
        return chunk_files, metadata_files
            
    except Exception as e:
        print(f"Error downloading files: {e}")
        return [], []

def create_merged_file(local_dir, output_filename, filter_correct_only=False):

    output_path = os.path.join(local_dir, output_filename)
    
    total_samples = 0
    kept_samples = 0
    
    with gzip.open(output_path, 'wb', compresslevel=9) as fout:
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
                        
                        # Apply filter if requested
                        if filter_correct_only:
                            if sample.get('answer_correct') == 'yes':
                                pickle.dump(sample, fout, protocol=5)
                                kept_samples += 1
                        else:
                            pickle.dump(sample, fout, protocol=5)
                            kept_samples += 1
                        
                        if total_samples % 100 == 0:
                            print(f"  Processed {total_samples} samples, kept {kept_samples}")
                            
                except Exception as e:
                    print(f"Error processing {chunk_file}: {e}")
                    continue
        
        pickle.dump({'_end': True, 'num_samples': kept_samples}, fout, protocol=5)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Samples kept: {kept_samples}")
    print(f"File saved to: {output_path}")
    
    return output_path, kept_samples

def extract_question_from_prompt(full_user_prompt):
    """
    Extract just the question from full_user_prompt that contains system prompt
    """
    if isinstance(full_user_prompt, list):
        prompt_text = full_user_prompt[0]
    else:
        prompt_text = full_user_prompt
    
    if "<｜User｜>" in prompt_text:
        #split and get part after <｜User｜>
        user_part = prompt_text.split("<｜User｜>", 1)[1]
        #remove <｜Assistant｜> part if exists
        if "<｜Assistant｜>" in user_part:
            question = user_part.split("<｜Assistant｜>", 1)[0].strip()
        else:
            question = user_part.strip()
        return question
    else:
        #return the full prompt if format is unexpected
        return prompt_text

def get_chunk_questions(local_dir):
    """
    Extract all questions from chunks to compare with GSM8K
    """
    gsm8k = load_dataset("gsm8k", "main")
    gsm8k_questions = set(sample['question'] for sample in gsm8k['train'])
    
    questions = set()
    not_found_questions = []
    
    chunk_files_local = sorted([f for f in os.listdir(local_dir) if f.startswith("chunk_") and f.endswith(".pkl.gz")])
    
    for chunk_file in chunk_files_local:
        chunk_path = os.path.join(local_dir, chunk_file)
        
        with gzip.open(chunk_path, "rb") as fin:
            try:
                chunk_header = pickle.load(fin)
                samples_in_chunk = chunk_header['num_samples']
                
                for i in range(samples_in_chunk):
                    sample = pickle.load(fin)
                    question = extract_question_from_prompt(sample['full_user_prompt'])
                    questions.add(question)
                    
                    # Validate that this question exists in GSM8K
                    if question not in gsm8k_questions:
                        not_found_questions.append({
                            'question': question[:200] + "..." if len(question) > 200 else question,
                            'chunk_file': chunk_file,
                            'sample_index': i
                        })
                    
                    if len(questions) <= 3:  # Debug first few
                        print(f"Debug - extracted question: {question[:100]}...")
                        print(f"  Found in GSM8K: {question in gsm8k_questions}")
                    
            except Exception as e:
                print(f"Error processing {chunk_file}: {e}")
                continue
    
    print(f"Found {len(questions)} unique questions in chunks")
    print(f"Questions matching GSM8K: {len(questions) - len(not_found_questions)}")
    print(f"Questions NOT found in GSM8K: {len(not_found_questions)}")
    
    if not_found_questions:
        print(f"\nWARNING: {len(not_found_questions)} questions from chunks not found in GSM8K!")
        print("First few examples:")
        for i, nfq in enumerate(not_found_questions[:5]):
            print(f"  {i+1}. {nfq['question']} (from {nfq['chunk_file']})")
        
        if len(not_found_questions) > 5:
            print(f"  ... and {len(not_found_questions) - 5} more")
        
    else:
        print("All extracted questions found in GSM8K training set")
    
    return questions

def create_filtered_gsm8k(chunk_questions, output_filename="gsm8k_filtered.pkl.gz"):

    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k['train']
    
    print(f"Original GSM8K training set: {len(train_data)} samples")
    
    filtered_samples = []
    excluded_count = 0
    
    for sample in train_data:
        if sample['question'] not in chunk_questions:
            filtered_samples.append(sample)
        else:
            excluded_count += 1
    
    print(f"Filtered GSM8K: {len(filtered_samples)} samples (excluded {excluded_count})")
    
    # Save filtered GSM8K
    with gzip.open(output_filename, 'wb', compresslevel=9) as f:
        pickle.dump({
            'metadata': {
                'original_size': len(train_data),
                'filtered_size': len(filtered_samples),
                'excluded_count': excluded_count,
                'description': 'GSM8K training set with chunk questions removed'
            }
        }, f, protocol=5)
        
        for sample in filtered_samples:
            pickle.dump(sample, f, protocol=5)
        
        pickle.dump({'_end': True, 'num_samples': len(filtered_samples)}, f, protocol=5)
    
    return output_filename, len(filtered_samples)

def upload_to_hf(file_path, repo_id, filename):
    """
    Upload file to HuggingFace Hub
    """
    try:
        print(f"Uploading {filename} to {repo_id}...")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Successfully uploaded {filename}")
        return True
    except Exception as e:
        print(f"Error uploading {filename}: {e}")
        return False

def main():
    repo_id = "lizardp1/gsm8k_early_exit"
    local_dir = "download_teacher_data"
    
    chunk_files, metadata_files = download_chunks_from_hf(repo_id, local_dir)
    
    if not chunk_files:
        print("No chunks downloaded, exiting")
        return
    
    merged_all_path, total_samples = create_merged_file(
        local_dir, 
        "merged_all_samples.pkl.gz", 
        filter_correct_only=False
    )
    
    merged_correct_path, correct_samples = create_merged_file(
        local_dir, 
        "merged_correct_only.pkl.gz", 
        filter_correct_only=True
    )
    
    chunk_questions = get_chunk_questions(local_dir)
    #gsm8k_filtered_path, gsm8k_remaining = create_filtered_gsm8k(chunk_questions, local_dir)
    gsm8k_filtered_path, gsm8k_remaining = create_filtered_gsm8k(chunk_questions, os.path.join(local_dir, "gsm8k_filtered.pkl.gz"))
    
    
    upload_results = []
    
    # Upload merged files
    upload_results.append(upload_to_hf(merged_all_path, repo_id, "merged_all_samples.pkl.gz"))
    upload_results.append(upload_to_hf(merged_correct_path, repo_id, "merged_correct_only.pkl.gz"))
    
    # Upload filtered GSM8K dataset
    if gsm8k_filtered_path and os.path.exists(os.path.join(local_dir, "gsm8k_filtered.pkl.gz")):
        upload_results.append(upload_to_hf(os.path.join(local_dir, "gsm8k_filtered.pkl.gz"), repo_id, "gsm8k_filtered.pkl.gz"))
    else:
        print("GSM8K filtered file not found, skipping upload")
        upload_results.append(False)
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total samples in chunks: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"GSM8K samples remaining after filtering: {gsm8k_remaining}")
    print(f"Upload success: {sum(upload_results)}/{len(upload_results)}")
    
    print(f"\nFiles uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()