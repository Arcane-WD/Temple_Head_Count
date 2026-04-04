import os

OUTPUT_FILE = "project_codebase.txt"
# Only aggregate relevant files for the project
TARGET_EXTENSIONS = ['.py', '.md', '.ts', '.tsx', '.json', '.js', '.jsx', '.css', '.yaml']

# Exclude directories
EXCLUDE_DIRS = {
    '.venv', 'venv', '__pycache__', '.git', 'yolo_dataset', 'runs', 
    'sampleio', 'assets', '.next', 'node_modules', 'data', 'input_vids', 
    'output_vids'
}

def aggregate_codebase():
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write("========================================================\n")
        outfile.write("TEMPLE PROJECT CODEBASE AGGREGATION\n")
        outfile.write("========================================================\n\n")
            
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in sorted(files):
                if any(file.endswith(ext) for ext in TARGET_EXTENSIONS):
                    # Skip the aggregator itself and the output file
                    if file == os.path.basename(__file__) or file == OUTPUT_FILE:
                        continue
                        
                    filepath = os.path.join(root, file)
                    outfile.write(f"\n\n\n{'='*80}\n")
                    outfile.write(f"FILE: {filepath}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                            count += 1
                    except Exception as e:
                        outfile.write(f"<<< Error reading file: {e} >>>\n")
                        
    print(f"Successfully aggregated {count} files into '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    aggregate_codebase()
