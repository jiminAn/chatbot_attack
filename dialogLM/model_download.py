import gdown

drive_path = 'https://drive.google.com/uc?id='
models = {'kogpt_wellness_epoch5_batch2.pth': '1zDZY9SXbyHsWM_nEsVTqYYf6hNWNQBWt'}

save_path = "./checkpoint/"

for name, id  in models.items():
    gdown.download(drive_path+id, save_path+name, quiet = False)
