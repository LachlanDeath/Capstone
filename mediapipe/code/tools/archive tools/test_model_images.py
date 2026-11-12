import os
import tkinter as tk
from tkinter import filedialog
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

def main():
    # 1. Initialize the GUI file picker (hidden main window)
    root = tk.Tk()
    root.withdraw()
    
    # Open dialog to select one or multiple image files
    print("Opening file dialog... Please select your image(s).")
    file_paths = filedialog.askopenfilenames(
        title="Select Images for Cello Posture Keypoints Workflow",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
    )
    
    if not file_paths:
        print("No files selected. Exiting script.")
        return

    print(f"Selected {len(file_paths)} file(s). Connecting to local Roboflow server...")

    # 2. Connect to your LOCAL server endpoint (bypasses serverless tier limitations)
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",  # Pointing to your local inference engine
        api_key="lpm4o3wO2VNKjsZNUXho"
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))

    # 3. Process each selected image
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"\n--- Processing: {filename} ---")
        
        try:
            result = client.run_workflow(
                workspace_name="maxsloan2005-outlook-com",
                workflow_id="cello-posture-keypoints",
                images={
                    "image": path
                },
                use_cache=True  # Bypasses redownloading workflow JSON structures after first pull
            )
            
            # 4. Get and display your results
            print(f"Result for {filename}:")
            print(result)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
