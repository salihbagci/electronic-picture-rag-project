# Electronic Picture RAG Project (Elektronik Parça Görsel RAG)

Bu proje, bir elektronik parça fotoğrafı yükleyince parçanın adını tahmin eder.
Yöntem:
- CLIP ile görsel embedding çıkarılır
- FAISS ile en benzer görseller bulunur
- Güven düşükse 'unknown' döner

## Klasör Yapısı

electronic-picture-rag-project/
- data/images/           -> Etiketlere göre ayrılmış görseller
- index/                 -> FAISS index dosyaları (images.faiss, items.json)
- src/
  - config.py
  - build_index.py       -> Index üretir (1 kere çalıştırılır)
  - app.py               -> Gradio arayüzünü açar
- requirements.txt       -> Gerekli kütüphaneler
- README.md              -> Bu dosya

## Kurulum

1) Kütüphaneleri kur:
pip install -r requirements.txt

## Index Oluşturma (1 kere)

Eğer index klasöründe images.faiss ve items.json yoksa:
python src/build_index.py

## Uygulamayı Çalıştırma

python src/app.py

Sonra Gradio arayüzü açılır, resim yükleyerek test edebilirsin.

## Notlar

- Colab restart olunca silinmemesi için proje Google Drive altında tutuluyor.
- Alakasız resimler için unknown dönmesi: top-k oylama + skor eşiği ile yapılır.
