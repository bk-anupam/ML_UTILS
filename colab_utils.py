from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
my_drive = GoogleDrive(gauth)

def empty_gdrive_trash():
    deleted_file_name = []
    for a_file in my_drive.ListFile({'q': "trashed = true"}).GetList():
        file_name = a_file['title']
        deleted_file_name.append(file_name)
        # delete the file permanently.
        a_file.Delete()
    print("The below files were cleared from trash")
    print(deleted_file_name)