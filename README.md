# Twitter-Sentiment-Analysis

Bu proje, **Twitter verilerini kullanarak duygu analizi** yapmayı hedefleyen doğal dil işleme (NLP) tekniklerini içerir. 

---

## I. VERİ KEŞFİ VE ÖNİŞLEME
Bu adımda verinin genel yapısını inceledim. Verisetinde başlık ve sütun adları olmadığı için sütunları kendim adlandırdım. Ardından veriseti çok büyük olduğu için daha kolay çalışmak için verisetini 100.000 veriye indirdim. Bu adımı yapmadan önce tokenizasyon ve lemmatizasyon işlemlerinin tamamlanması çok uzun sürmüştü.

### Veri Önişleme Adımları:
**1. Text sütunu küçük harflere dönüştürme**

Büyük-küçük harf duyarlılığını ortadan kaldırdım. Örneğin, "Happy" ve "happy" aynı anlamda olduğundan, küçük harfe
dönüştürerek metindeki tutarlılığı sağladım.

**2. Tweet içindeki Url’leri kaldırma**

URL'ler genellikle model için yararlı bilgi içermez ve analiz sürecine gürültü katar.

**3. Kullanıcı adlarını kaldırma**

"@username" formatındaki kullanıcı adları model için anlamlı bilgi içermez ve metnin genel anlamını bozabilir.

**4. Hashtagleri kaldırma**

Hashtagler (#example) modelde anlamsız veya fazla ağırlık taşıyabilir. Metin daha temiz hale gelmesi ve modelin gürültüden etkilenmesini engelledim.

**5. Noktalama işaretlerini kaldırma**

Noktalama işaretleri genellikle duygu analizi için doğrudan bir bilgi içermez.

**6. Sayıları kaldırma**

Tweetlerde geçen sayılar genellikle duygu analizi için faydalı bir bilgi içermez. Gürültüyü engellemek adına sayıları
kaldırdım.

**7. Fazla boşluklarını tek bir boşluğa indirme**

Fazladan boşluklar veri temizliği sırasında oluşabilir ve bu durum analizi zorlaştırabilir. Metin, tek bir boşluk formatına
dönüştürülerek düzenli hale getirdim.

**8. Stop-words**

"the", "is", "and" gibi duygu analizi için anlam ifade etmeyen sık kullanılan kelimeleri çıkardım.

**9. Spacy modeli ile tokenizasyon ve lemmatizasyon**

Tokenizasyon: Cümleleri, tek tek kelimelere (token) bölerek işleme hazır hale getirdim.
Lemmatizasyon: Kelimeleri kök formlarına dönüştürerek, örneğin "running" -> "run", "better" -> "good" şeklinde normalize ettim.

## II. ÖZELLİK ÇIKARIMI VE MODELLEME
Bu adımda, metin verilerini sınıflandırmak için TF-IDF yöntemini kullanarak özellik çıkarımı yaptım ve ardından Logistic Regression modeli eğittim. Modelin performansını, Precision, Recall, F1-score ve Accuracy gibi değerlendirme metrikleri kullanılarak ölçtüm. 

TF-IDF vektörleştirme yöntemi, metinlerdeki kelimelerin sıklığını ve kelimenin belgedeki özgüllüğünü göz önünde bulundurarak her bir kelimeye bir ağırlık verir. 

Sınıflandırma modelinin daha verimli çalışmasını sağlamak için ilk olarak, verilen veri setinden text ve polarity sütunlarını ayırdım. Text sütunu metin verilerini, polarity sütunu ise metinlerin duygu durumlarını (pozitif veya negatif) temsil etmektedir. Polarity etiketlerini, duygu durumlarına göre düzenledim. 

Eğitim ve test verileri, train_test_split fonksiyonu ile %80 eğitim ve %20 test olarak ayırdım. Modelin parametreleri arasında max_iter=300 kullanılarak maksimum 300 iterasyonla modelin eğitilmesi sağlanmıştır. Bu parametre, modelin öğrenme sürecinde doğruluğu arttırabilir.

## III. TRANSFORMER TABANLI DİL MODELLERİ
Bu adımda, Hugging Face Transformers kütüphanesini kullanarak bir DistilBERT modelini ince ayar (finetuning)
yapmaya çalıştım. Veri setini, Hugging Face'in Dataset formatına dönüştürerek eğitim (train) ve test verileri olarak ayırdım. 

DistilBERT modelinin tokenizer ve model yapılarını kullandım. DistilBertTokenizer metin verilerini uygun formata dönüştürürken, DistilBertForSequenceClassification sınıflandırma modelini kullanarak modelin num_labels parametresi 2 olarak ayarladım, bu da ikili sınıflandırmayı işaret eder. 

Veriler üzerinde tokenizasyon işlemi yaptım. Her bir metini, DistilBERT tokenizer kullanılarak uygun formatta (max_length parametresi ile uzunluk sınırlaması ve truncation ile kesme) dönüştürdüm. 

Eğitim sürecinde kullanılan parametreler:

**1. learning_rate:** 

Modelin öğrenme oranı (adım aralığı). 

**2. per_device_train_batch_size:**

Eğitimde kullanılacak batch boyutu.

**3. num_train_epochs:**

Eğitim süresi (epoch sayısı) 1 olarak
ayarlandı.

**4. logging_steps:**

Eğitimdeki logların hangi sıklıkla kaydedileceğini belirledi.

Bu model için tahmin olarak bu iki cümleyi verdiğimde:

*"I love this product!"* — Bu cümle için tahmin 1. Bu, modelin cümlenin pozitif bir duygu taşıdığını doğru şekilde anladığını gösterir.

*"This is the worst experience ever."* — Bu cümle için tahmin 0. Model, cümlenin olumsuz bir duygu içerdiğini doğru şekilde tespit etmiş.

## IV. SONUÇLARI KARŞILAŞTIRMA
Her iki modelin başarısı, Confusion Matrix (karmaşıklık matrisi) ile görselleştirilmiştir. Bu görsellere bakıldığında bu
karşılaştırmalar yapılabilir:

*Doğru Tahminler (True Positives ve True Negatives)*

Her iki model de doğru tahminlerde yüksek başarı göstermektedir. Ancak, BERT modelinin doğru negatiflerde (7971) ve doğru pozitiflerde (7770) daha yüksek performans sergilediği görülmektedir. Logistic Regression modelinin doğru tahminleri ise sırasıyla 7378 (negatif) ve 7738 (pozitif) olarak daha düşüktür.

*Yanlış Tahminler (False Positives ve False Negatives)*

Logistic Regression modelinde yanlış pozitiflerin (2617) ve yanlış negatiflerin (2267) sayısı BERT modeline kıyasla daha yüksektir. BERT, her iki türde de daha düşük hata oranı sergileyerek daha iyi genel doğruluk sağlamaktadır. 

BERT modeli, hem doğru tahminlerde hem de yanlış tahminlerde daha iyi performans göstermektedir. Özellikle büyük veri setleriyle çalışan Transformer tabanlı modellerin, Logistic Regression gibi temel yöntemlere kıyasla daha iyi sonuçlar verdiği söylenebilir. Logistic Regression ise daha hızlı ve basit bir model olmasına rağmen, karmaşık dil modelleme görevlerinde sınırlı kalabilir. BERT, dil anlayışı ve bağlamı daha derinlemesine öğrenebildiği için daha doğru tahminler yapmaktadır. BERT, özellikle dil tabanlı görevlerde daha etkili ve doğru sonuçlar üretirken, Logistic Regression modelinin ise daha basit ve daha az güçlü bir yaklaşım sunduğu görülmektedir.

| Özellik  | Logistic Regression  | BERT  |
| --|:-------:| -----:|
| Modelin Karmaşıklığı| Basit ve doğrusal  | Derin öğrenme ve Transformer tabanlı|
| Veri İhtiyacı |Küçük ve orta büyüklükte veri setleri için uygun   | Büyük veri setlerinde daha iyi performans    |
| Hız |Hızlı eğitim ve tahmin süresi     | Yavaş eğitim ve tahmin süresi  |
| Performans | Basit görevlerde yeterli, büyük veri setlerinde düşük doğruluk     | Yüksek doğruluk, karmaşık görevlerde güçlü  |
| Yorumlanabilirlik | Kolayca yorumlanabilir     | Zor yorumlanabilir  |
| Karmaşıklık | Düşük     | Yüksek |
| Hesaplama Maliyeti| Düşük hesaplama gücü gerektirir     | Yüksek hesaplama gücü ve bellek kullanımı  |
