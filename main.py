import cv2
import imutils
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import re
import os
import json
import pymongo
from urllib.request import urlopen
from datetime import datetime


#%% -----------------------Database Balantisi-----------------------------------

# MongoDB Balantisi
client = pymongo.MongoClient("mongodb://localhost:27017/")  
database_name = "find-car-location"  
db = client[database_name]
collection_name = "plaques"

# Verileri çek
plaques_collection = db[collection_name]
db_plaques = db[collection_name].find()
print(type(db_plaques))

#%%-----------------------Lokasyon ve tarih ----------------------------------------------------

# aracın bulunduğu konumu bulmak için fonksiyon.
def getLocation () :
    url = "https://ipinfo.io/json"
    response = urlopen(url)
    data = json.load(response)
    location = data['loc']
    return location

# random lokasyon oluşturma, test ortamı için
import random
def random_location():
    return f'{random.uniform(0, 100):.4f},{random.uniform(0, 100):.4f}'


# konum bulunduğunda alınan tarih ve saat
def getDate() :
    date = datetime.now()
    custom_date = date.strftime("%Y-%m-%d %H:%M:%S")

    print(custom_date)
    return custom_date

#%% --------------------------görüntü işleme ve ocr-----------------------------------

path_name = os.listdir('example_image')

for i in path_name :
    images = os.path.join('example_image', i)

    img = cv2.imread(images)

    img = cv2.resize(img, (800, 500))

    plt.imshow(img, cmap="gray")
    plt.title('original resim')
    plt.show()

    # Resmi alıp griye çeviriyor.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap="gray") #
    plt.title('gri resim')
    plt.show()

    gray = cv2.bilateralFilter(gray, 13, 15, 15) # (source_image, diameter of pixel, sigmaColor, sigmaSpace)
    

    plt.imshow(gray, cmap="gray") # filtre uygulanmış resim
    plt.title('filtre uygulanmis')
    plt.show()

    # cany edge kenar algılama algoritması
    cany = cv2.Canny(gray, 30, 200)  # 30, 200 yoğunluk min ve max

    plt.imshow(cany, cmap="gray") # filtre uygulanmış resim
    plt.title('canny uygulanmis')
    plt.show()

    # Kontür alma
    contours = cv2.findContours(cany.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    # Tersten sıralama yaptırıyoruz ve ilk 10 değeri çekiyoruz.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = 0

    for c in contours:  

        _ = cv2.arcLength(c, True)
    
        approx = cv2.approxPolyDP(c, 0.018 * _, True)
        # Daha düzgün dikdörtgen algılatmak
        # Dörtgen bir cisim
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
    else:
        detected = 1  # Eğer varsa

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)  # Kontür çizilir.

        # bulunan plakayı çiz
        (x, y, w, h) = cv2.boundingRect(screenCnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # fotografta bulunan plakayı göster
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Bulunan Plaka')
    plt.show()

    # Numara plakası dışındaki kısmı maskeleme
    mask = np.zeros(gray.shape, np.uint8)
    # shape = satır sütun piksel sayısı ve renkli ise renk katmanları

    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # algılanan resmin dışındakileri siyah yapar


    (x, y) = np.where(mask == 255)
    # np.where(mask<5,-1,100) şeklinde çalışsaydı eğer
    # koşulu sağlamadığında -1 değerini atayacak sağladığında 100
    # np.where(mask==255) olduğunda ise true false döndürecek

    (tx, ty) = (np.min(x), np.min(y))  # Dizinin minimum ve maksimum elemanını buluyor.
    (bx, by) = (np.max(x), np.max(y))
    Cropped = gray[tx:bx + 1, ty:by + 1]  # tx = bx + 1 kadar, ty=by +1 kadar
    custom_config = r'-l eng --psm 6'  # config l dil kodu --psm


    Cropped = cv2.resize(Cropped, (300,100))
    cv2.imshow('plate' , Cropped)


    #%% Easyocr Modeli 

    reader = easyocr.Reader(['tr'])
    text = reader.readtext(Cropped, detail= 0, paragraph=True)

    found = text[0]
    found_new = found.replace(" ", "")
    print(found_new)

    

    #%% bulunan plaka varsa Db yazma
    plaques_collection = db[collection_name]
    db_plaques = db[collection_name].find()

    found_plaques = found_new  # bulunan trafik plakası 
    print("found_plaques", found_plaques)


    # found_location = getLocation() # canlı ortam

    found_location = random_location() # test ortamı

    found_date = getDate() 
    

    for data in db_plaques : # bulunan trafik plakası, veri tabanında var mı ? var ise bu plakayı' ve konumunu car_location_db aktarıyoruz.
        print(data["foundLocation"])
        print(data["plaqueName"])
        #print(data["dateAndClock"])
        if(data["plaqueName"] == found_plaques) : 
                    

            data["foundLocation"] = found_location
            data["dateAndClock"] = found_date

            plaques_collection.update_one(
                {"plaqueName": found_plaques},
                {
                    "$set": {
                        "foundLocation": found_location,
                        "dateAndClock": found_date
                    }
                }
            )


    img = cv2.resize(img, (500, 300))
    Cropped = cv2.resize(Cropped, (200, 70))


    cv2.waitKey(0)
    cv2.destroyAllWindows()





'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr


# foto oku
img = cv2.imread('example_image/2.jpg')
cv2.resize(img, (500,500))

plt.imshow(img)
plt.show()

# foto gri çevir
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap="gray")
plt.show()

#filtreleme-blurlama
filter_img = cv2.bilateralFilter(gray_img, 4,100,100)
plt.imshow(filter_img, cmap="gray")
plt.show()


# kenar algılama
cany_img = cv2.Canny(filter_img, 50, 150)
plt.imshow(cany_img, cmap="gray")
plt.show()

# Konturları bul
contours, _ = cv2.findContours(cany_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# En büyük konturu bul
largest_contour = max(contours, key=cv2.contourArea)

# Konturu çevreleyen sınırlayıcı dikdörtgenin koordinatlarını hesapla
x, y, w, h = cv2.boundingRect(largest_contour)
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
plt.imshow(img), plt.show()


# Belirtilen alandaki bölgeyi kes
cropped_region = img[y:y+h, x:x+w]


# Kesilen bölgeyi bir dosyaya kaydet
cv2.imwrite('found_plaque.jpg', cropped_region)

# Kesilen bölgeyi göster
cv2.imshow('Kesilen Alan', cropped_region)

# OCR Model
reader = easyocr.Reader(['en'])

result = reader.readtext(cropped_region)
for detection in result:
    print(detection[1])  # Metni ekrana yazdır


cv2.waitKey(0)
cv2.destroyAllWindows()




'''













