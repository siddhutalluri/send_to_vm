import os

def rename_images(folder_path, folder_name):
    for i, filename in enumerate(os.listdir(folder_path)):
        old_path = os.path.join(folder_path, filename)
        extension = os.path.splitext(filename)[1]  # Get the file extension
        new_filename = f'{folder_name}{i}{extension}'
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} to {new_path}')

# Example usage:
folder_path = 'watch_data/Steel ring removed'
folder_name = 'steel_ring_removed'

rename_images(folder_path, folder_name)
