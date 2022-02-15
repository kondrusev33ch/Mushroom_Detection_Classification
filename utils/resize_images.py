from PIL import Image
from tqdm import tqdm
import os


def resize(f_path: str, t_path: str, dirs):
    # Check for preprocessed folder
    if not os.path.exists(t_path):
        os.makedirs(t_path)

    for i, item in enumerate(tqdm(dirs)):
        if os.path.isfile(f_path + item):
            with Image.open(f_path + item) as img:
                # Resize image
                img_resized = img.resize((200, 200), Image.ANTIALIAS)

                # In case you have different types of images
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')

                # Save resized image
                img_resized.save(t_path + f'{i:04d}.jpg', 'JPEG', quality=90)


if __name__ == '__main__':
    from_path = 'data/Detection/'
    to_path = from_path + 'preprocessed/'
    directories = os.listdir(from_path)

    resize(from_path, to_path, directories)
