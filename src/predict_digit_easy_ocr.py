import easyocr
from PIL import Image
import re


def predict_single_image(image_path):
    reader = easyocr.Reader(["en"])

    try:
        result = reader.readtext(image_path)
        detected_text = " ".join([entry[1] for entry in result])

        numbers = re.findall(r"\d+", detected_text)
        processed_numbers = []

        for num in numbers:
            if len(num) == 1:
                processed_numbers.append(num)
            elif len(num) == 2 and "1" in num:
                # Keep only the non-1 digit
                other_digit = next(d for d in num if d != "1")
                processed_numbers.append(other_digit)

        return " ".join(processed_numbers)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


# def main():
#     image_path = "/content/Screenshot 2024-12-11 at 14.51.26.png"
#     text = predict_single_image(image_path)
#     print(text)
#     if text:
#         print(f"Detected numbers: {text}")

# if __name__ == "__main__":
#     main()
