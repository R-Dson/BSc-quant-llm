import os
import json

def find_json_files_with_result_in_subfolders():
    """Find all JSON files in subfolders one level down with 'result' in their name."""
    json_files = {}
    for root, dirs, files in os.walk('.', topdown=True):
        # Only go one level down
        if root.count(os.sep) - os.getcwd().count(os.sep) >= 1:
            continue
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            subdir_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.json') and 'result' in f]
            if subdir_files:
                json_files[subdir] = subdir_files
    return json_files

def get_accuracy_from_json(file_name):
    """Extract the accuracy value from the given JSON file."""
    with open(file_name, 'r') as file:
        data = json.load(file)
        return data.get('accuracy')

def main():
    json_files_in_subfolders = find_json_files_with_result_in_subfolders()
    for subdir, json_files in json_files_in_subfolders.items():
        if json_files:
            total_accuracy = 0
            count = 0
            print(f"Subfolder: {subdir}")
            for json_file in json_files:
                accuracy = get_accuracy_from_json(json_file)
                if accuracy is not None:
                    accuracy_percent = accuracy * 100  # Convert to percentage
                    total_accuracy += accuracy
                    count += 1
                    print(f"  File: {os.path.basename(json_file)}, Accuracy: {accuracy_percent:.2f}")
                else:
                    print(f"  File: {os.path.basename(json_file)} does not contain 'accuracy' tag")
            if count > 0:
                average_accuracy = (total_accuracy / count) * 100  # Calculate average accuracy as percentage
                print(f"  Average Accuracy: {average_accuracy:.2f}")
            else:
                print(f"  No files with 'accuracy' tag found")
        else:
            print(f"Subfolder: {subdir} has no relevant JSON files")

if __name__ == "__main__":
    main()
