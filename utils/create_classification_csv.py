import pandas as pd
import glob
from PIL import Image
import os

# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Converts every image to .jpg ans save to the same directory
    for filename in glob.glob('data/Classification/train/*'):
        if filename.lower().endswith(('.png', '.jpeg', '.tiff', '.bmp', '.gif')):
            with Image.open(filename) as image:
                image.convert('RGB').save(filename[:-4] + '.jpg')
            # os.remove(filename)  # in case you want to delete old images

    # Create training dataframe
    img_ids = []
    classes = []
    for filename in glob.glob('data/Classification/train/*.jpg'):
        filename = os.path.basename(filename)[:-4]
        img_ids.append(filename)
        classes.append(filename[-4:])

    train_df = pd.DataFrame(data={'image_id': img_ids, 'mushroom': classes})
    print(train_df)

    # Save dataframe to csv
    train_df.to_csv('data/Classification/train.csv', index=False)
