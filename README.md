# Bottleneck Detection Application

Bu proje, veri setleri üzerinden **darboğaz (bottleneck) tespiti** yapılmasını amaçlayan bir makine öğrenmesi tabanlı analiz ve web uygulamasıdır. Proje kapsamında eğitilmiş bir derin öğrenme modeli kullanılarak, yüklenen veriler üzerinde otomatik analiz gerçekleştirilmekte ve sonuçlar kullanıcıya web arayüzü üzerinden sunulmaktadır.

Uygulama, **Streamlit Cloud** üzerinde canlı olarak çalışmaktadır ve GitHub reposu ile entegre edilmiştir.

---

## 🎯 Projenin Amacı

Bu projenin temel amaçları şunlardır:

* Veri setleri üzerinde darboğaz oluşturan durumların otomatik olarak tespit edilmesi
* Makine öğrenmesi modelinin gerçek zamanlı olarak kullanılması
* Analiz sürecinin kullanıcı dostu bir web arayüzü ile sunulması
* Akademik bir proje kapsamında uçtan uca bir ML uygulaması geliştirilmesi

---

## 🧠 Kullanılan Teknolojiler

* **Python**
* **PyTorch** – Derin öğrenme modeli
* **Transformers** – Model altyapısı
* **Streamlit** – Web arayüzü
* **Pandas / NumPy** – Veri işleme
* **Jupyter Notebook** – Model eğitimi ve analiz

---

## 📁 Proje Dosya Yapısı

```
Bottleneck-Detection/
│
├── app.py
│   Streamlit tabanlı web uygulaması
│
├── best_model.pt
│   Eğitilmiş makine öğrenmesi modeli (Git LFS ile yönetilmektedir)
│
├── bottleneck_detection.ipynb
│   Model eğitimi ve veri analizi adımlarını içeren Jupyter Notebook
│
├── requirements.txt
│   Projede kullanılan Python kütüphaneleri
│
└── README.md
│   Proje dokümantasyonu
```

---

## ▶️ Uygulamanın Çalıştırılması (Yerel)

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Streamlit uygulamasını başlatın:

```bash
streamlit run app.py
```

---

## ☁️ Canlı Uygulama (Streamlit Cloud)

Uygulama Streamlit Cloud üzerinde canlı olarak çalışmaktadır:

🔗 **Live Demo:**
*https://bottleneck-detection-ai.streamlit.app/*

---

## 📊 Notebook Açıklaması

`bottleneck_detection.ipynb` dosyasında:

* Veri setinin incelenmesi
* Ön işleme adımları
* Model eğitimi
* Model performans değerlendirmeleri

ayrıntılı şekilde yer almaktadır. Bu dosya, uygulamanın arka planındaki akademik ve teknik süreci belgelemek amacıyla repoda tutulmaktadır.

---

## ⚠️ Model Dosyası Hakkında Önemli Not

> Eğitilmiş model dosyası (`best_model.pt`) büyük boyutlu olduğu için **Git LFS** kullanılarak repoya eklenmiştir. Bu sayede Streamlit Cloud ortamında herhangi bir kod değişikliği yapılmadan model doğrudan kullanılabilmektedir.

---

## 👩‍💻 Geliştirici

**Sevgi Akyüz**
Bu proje, akademik bir sunum ve kişisel portföy çalışması kapsamında geliştirilmiştir.

---

## 📌 Notlar

* Streamlit Cloud üzerinde `.ipynb` dosyaları çalıştırılmaz
* Web uygulamasının ana giriş noktası `app.py` dosyasıdır
* Model, uygulama başlatılırken otomatik olarak yüklenmektedir

