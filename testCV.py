import cv2
import pytesseract
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    #(1 frame every 0.5 seconds)
    frame_rate = 2  
    frame_count = 0
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_directory}/frame_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()



def read_digital_display(image_path):
    """Reads digits from a digital display image."""
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize image if it's too small or too large
    height = 300  # target height
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    image = cv2.resize(image, (width, height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    # Clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Configure Tesseract specifically for digital display
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    
    # Read text
    text = pytesseract.image_to_string(cleaned, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:PM')
    
    # Clean up the result
    text = text.strip()
    
    # Display the processed image (for debugging)
    #cv2.imshow('Processed Image', cleaned)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return text



def process_video_frames(folder_path):
    # Get all files in the folder with their full paths
    files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(('.jpg', '.png', '.jpeg'))  # Valid image extensions
    ]
    
    # Sort the files by their creation time
    files.sort(key=os.path.getctime)
    
    # Iterate through the sorted files
    for image_path in files:
        filename = os.path.basename(image_path)  # Extract filename from the full path
        
        # Call the read_digital_display function and print the result
        try:
            time = read_digital_display(image_path)
            print(f"Detected time in {filename}: {time}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")



if __name__ == "__main__":

    video_file = r"vid.mp4"  
    extract_frames(video_file)

    folder_path = "vid_frames"
    process_video_frames(folder_path)