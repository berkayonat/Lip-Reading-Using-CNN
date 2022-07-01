import numpy as np
import os
from PIL import Image

def concatFrames(path1,path2):
    def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                          for im in im_list]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new('RGB', (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                          for im in im_list]
        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst
        
        
    def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
        im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
        return get_concat_v_multi_resize(im_list_v, resample=resample)

    ima = []

    l = 1
    while l < 30:
        
        if l < 10:
              path = path1 +'/color_00' +str(l)
        else:
              path = path1 +'/color_0' +str(l)
        
        
        imagep2 = ".jpg"
        imagePath = path + imagep2            
        
            
        try:
            image = Image.open(imagePath)

            if np.any(image != None) :
                ima.append(image)
        except: 
            break
                        
        l=l+1

    if (len(ima) == 4):
        get_concat_tile_resize([[ima[0], ima[1]],
                                [ima[2], ima[3]]]).save(path2+'.jpg')
        
        
    elif (len(ima) == 5):
        get_concat_tile_resize([[ima[0], ima[1], ima[2]],
                                [ima[3], ima[4], ima[4]]]).save(path2+'.jpg')


    elif (len(ima) == 6):
        get_concat_tile_resize([[ima[0], ima[1], ima[2]],
                                [ima[3], ima[4], ima[5]]]).save(path2+'.jpg')


    elif (len(ima) == 7):
        get_concat_tile_resize([[ima[0], ima[1], ima[2]],
                                [ima[3], ima[4], ima[5]],
                                [ima[6], ima[6], ima[6]]]).save(path2+'.jpg')


    elif (len(ima) == 8):
        get_concat_tile_resize([[ima[0], ima[1], ima[2]],
                                [ima[3], ima[4], ima[5]],
                                [ima[6], ima[7], ima[7]]]).save(path2+'.jpg')


    elif (len(ima) == 9):
        get_concat_tile_resize([[ima[0], ima[1], ima[2]],
                                [ima[3], ima[4], ima[5]],
                                [ima[6], ima[7], ima[8]]]).save(path2+'.jpg')


    elif (len(ima) == 10):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                                [ima[4], ima[5], ima[6], ima[7]],
                                [ima[8], ima[9], ima[9], ima[9]]]).save(path2+'.jpg')        
                        
                        
    elif (len(ima) == 11):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                                [ima[4], ima[5], ima[6], ima[7]],
                                [ima[8], ima[9], ima[10], ima[10]]]).save(path2+'.jpg')            
                
                    
    elif (len(ima) == 12):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                               [ima[4], ima[5], ima[6], ima[7]],
                               [ima[8], ima[9], ima[10], ima[11]]]).save(path2+'.jpg')
                                           
    elif (len(ima) == 13):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                               [ima[4], ima[5], ima[6], ima[7]],
                               [ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[12], ima[12], ima[12]]]).save(path2+'.jpg')
                                           
    elif (len(ima) == 14):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                               [ima[4], ima[5], ima[6], ima[7]],
                               [ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[13], ima[13]]]).save(path2+'.jpg')


    elif (len(ima) == 15):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                               [ima[4], ima[5], ima[6], ima[7]],
                               [ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[14]]]).save(path2+'.jpg')


    elif (len(ima) == 16):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3]],
                               [ima[4], ima[5], ima[6], ima[7]],
                               [ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15]]]).save(path2+'.jpg')

    elif (len(ima) == 17):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[16], ima[16], ima[16]]]).save(path2+'.jpg')


    elif (len(ima) == 18):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[17], ima[17]]]).save(path2+'.jpg')


    elif (len(ima) == 19):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[18]]]).save(path2+'.jpg')


    elif (len(ima) == 20):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]]]).save(path2+'.jpg')
                   
                                           
    elif (len(ima) == 21):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]],
                               [ima[20], ima[20], ima[20], ima[20], ima[20]]]).save(path2+'.jpg')


    elif (len(ima) == 22):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]],
                               [ima[20], ima[21], ima[21], ima[21], ima[21]]]).save(path2+'.jpg')


    elif (len(ima) == 23):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]],
                               [ima[20], ima[21], ima[22], ima[22], ima[22]]]).save(path2+'.jpg')


    elif (len(ima) == 24):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]],
                               [ima[20], ima[21], ima[22], ima[23], ima[23]]]).save(path2+'.jpg')


    elif (len(ima) == 25):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4]],
                               [ima[5], ima[6], ima[7], ima[8], ima[9]],
                               [ima[10], ima[11], ima[12], ima[13], ima[14]],
                               [ima[15], ima[16], ima[17], ima[18], ima[19]],
                               [ima[20], ima[21], ima[22], ima[23], ima[24]]]).save(path2+'.jpg')


    elif (len(ima) == 26):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4], ima[5]],
                               [ima[6], ima[7], ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15], ima[16], ima[17]],
                               [ima[18], ima[19], ima[20], ima[21], ima[22], ima[23]],
                               [ima[24], ima[25], ima[25], ima[25], ima[25], ima[25]]]).save(path2+'.jpg')


    elif (len(ima) == 27):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4], ima[5]],
                               [ima[6], ima[7], ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15], ima[16], ima[17]],
                               [ima[18], ima[19], ima[20], ima[21], ima[22], ima[23]],
                               [ima[24], ima[25], ima[26], ima[26], ima[26], ima[26]]]).save(path2+'.jpg')


    elif (len(ima) == 28):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4], ima[5]],
                               [ima[6], ima[7], ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15], ima[16], ima[17]],
                               [ima[18], ima[19], ima[20], ima[21], ima[22], ima[23]],
                               [ima[24], ima[25], ima[26], ima[27], ima[27], ima[27]]]).save(path2+'.jpg')


    elif (len(ima) == 29):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4], ima[5]],
                               [ima[6], ima[7], ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15], ima[16], ima[17]],
                               [ima[18], ima[19], ima[20], ima[21], ima[22], ima[23]],
                               [ima[24], ima[25], ima[26], ima[27], ima[28], ima[28]]]).save(path2+'.jpg')
        
        
    elif (len(ima) == 30):
        get_concat_tile_resize([[ima[0], ima[1], ima[2], ima[3], ima[4], ima[5]],
                               [ima[6], ima[7], ima[8], ima[9], ima[10], ima[11]],
                               [ima[12], ima[13], ima[14], ima[15], ima[16], ima[17]],
                               [ima[18], ima[19], ima[20], ima[21], ima[22], ima[23]],
                               [ima[24], ima[25], ima[26], ima[27], ima[28], ima[29]]]).save(path2+'.jpg')
        
        
        
speakers = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08'] 
word_folder = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
varieties = ['01','02','03','04','05','06','07','08', '09', '10']


os.mkdir('mouth3')
for speaker in speakers:
   os.mkdir('mouth3/'+speaker)
   
   for ind, folder in enumerate(word_folder):
     os.mkdir('mouth3/'+speaker+'/' +folder)
     
     for vari in varieties:
       os.mkdir('mouth3/'+speaker+'/'+folder+'/'+vari)
       path1 = 'mouth2/'+speaker+'/'+folder+'/'+vari  
       path2 = 'mouth3/'+speaker+'/'+folder+'/'+vari +'/'+'image'
       concatFrames(path1,path2)