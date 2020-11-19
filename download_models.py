import os

from google_drive_downloader import GoogleDriveDownloader as gdd

PATH = os.path.dirname(os.path.realpath(__file__))

print('Downloading pre-trained command model...')
gdd.download_file_from_google_drive(file_id='1S9nL8pC9Jgwfv1qEHjor2k2BWprMnsMu',
                                    dest_path=f'{PATH}\\command_model.pth',
                                    showsize=True)
