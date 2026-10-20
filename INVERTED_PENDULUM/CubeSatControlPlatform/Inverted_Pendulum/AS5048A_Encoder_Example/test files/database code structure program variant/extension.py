import time

# Function to read angle from file
def read_angle_from_file(file_path="angle_data.txt"):
    with open(file_path, "r") as file:
        while True:
            file.seek(0)  # Go to the beginning of the file
            angle = file.read().strip()
            if angle:
                print(f"Read Angle: {angle} degrees")
            time.sleep(0.01)  # Adjust sleep time as needed

if __name__ == "__main__":
    read_angle_from_file()
