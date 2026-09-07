import os
import urllib.request

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

FONTS = [
    ("https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf", "Montserrat-Black.ttf"),
    ("https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Bold.ttf", "Montserrat-Bold.ttf"),
    ("https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Regular.ttf", "Montserrat-Regular.ttf"),
    ("https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf", "Anton-Regular.ttf"),
    ("https://raw.githubusercontent.com/googlefonts/OswaldFont/master/fonts/ttf/Oswald-Regular.ttf", "Oswald-Regular.ttf"),
    ("https://raw.githubusercontent.com/googlefonts/OswaldFont/master/fonts/ttf/Oswald-Bold.ttf", "Oswald-Bold.ttf"),
]

def main():
    os.makedirs(FONTS_DIR, exist_ok=True)
    print(f"Downloading fonts to {FONTS_DIR}...")
    for url, filename in FONTS:
        filepath = os.path.join(FONTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists, skipping.")
    print("Done!")

if __name__ == "__main__":
    main()
