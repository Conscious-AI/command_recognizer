from google_drive_downloader import GoogleDriveDownloader as gdd

print('Downloading command model...')
gdd.download_file_from_google_drive(file_id='16C3cUHuGo3M2PpZV9Cwjp8aFaITDGPA7',
                                    dest_path='./command_model.pth',
                                    showsize=True)

print('Downloading librispeech model...')
gdd.download_file_from_google_drive(file_id='1zOp0PdMttsmLXZ2z556Np_VFYkW1Xdpk',
                                    dest_path='./librispeech_model.pth',
                                    showsize=True)