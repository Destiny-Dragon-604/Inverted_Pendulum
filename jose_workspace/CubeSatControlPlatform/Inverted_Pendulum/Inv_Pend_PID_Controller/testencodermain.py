import time
from AS5048AEncoder import AS5048AEncoder  # Replace with your actual filename

def test_encoder():
    try:
        # Create an instance of the encoder
        print("Initializing encoder...")
        encoder = AS5048AEncoder()
        print("Encoder initialized.")

        print("Initializing rest position...")
        time.sleep(2)  # Give some time to ensure the encoder is at rest
        encoder.rest_position = encoder.raw_to_degrees(encoder.read_raw_angle())  # Set initial rest position
        print(f"Rest position initialized at {encoder.rest_position:.2f} degrees.")

        # Continuously read and print the angle
        while True:
            try:
                angle = encoder.read_angle()
                print(f"Angle: {angle:.2f} degrees")
            except Exception as e:
                print(f"Error reading angle: {e}")

            time.sleep(encoder.reading_interval)

    except Exception as e:
        print(f"Failed to initialize encoder: {e}")
    finally:
        if 'encoder' in locals():
            print("Closing I2C bus.")
            encoder.close()

if __name__ == "__main__":
    test_encoder()

