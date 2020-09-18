import os

from google_drive_downloader import GoogleDriveDownloader as gdd

PATH = os.path.dirname(os.path.realpath(__file__))

print('Downloading command model...')
gdd.download_file_from_google_drive(file_id='16C3cUHuGo3M2PpZV9Cwjp8aFaITDGPA7',
                                    dest_path=f'{PATH}\\command_model.pth',
                                    showsize=True)

print('Downloading librispeech model...')
gdd.download_file_from_google_drive(file_id='1zOp0PdMttsmLXZ2z556Np_VFYkW1Xdpk',
                                    dest_path=f'{PATH}\\librispeech_model.pth',
                                    showsize=True)