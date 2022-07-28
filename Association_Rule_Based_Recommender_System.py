# Online Retail Association Rule Based Recommender System

# İş Problemi

""""
Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir.
Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız.
Ürün önerileri 1 tane ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.

Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

"""
# Veri Seti Hikayesi

"""
Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 
tarihleri arasındaki online satış işlemlerini içeriyor. 
Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

"""
# Proje Görevleri
# Görev 1
# Veriyi Hazırlama

# Adım 1
# Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.

!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/4_Tavsiye_Sistemleri/Association_Rule_Based_Recommender_System/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.isnull().sum()
df.describe().T

# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz.
# (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)

# Adım 3
# Boş değer içeren gözlem birimlerini drop ediniz.

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız.
# (C faturanın iptalini ifade etmektedir.)

# Adım 5
# Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST", na=False)]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T

# Görev 2
# Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme

# Adım 1
# Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.

df_ger = df[df['Country'] == "Germany"]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger)

ger_inv_pro_df = create_invoice_product_df(df_ger, id=True)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_ger, 10002)

# Adım 2
# Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.

def create_rules(dataframe):
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(ger_inv_pro_df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

# Görev 3
# Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma

# Adım 1
# check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_ger, 10002)

 # ['INFLATABLE POLITICAL GLOBE ']

# Adım 2
# arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

# Adım 3
# Önerilecek ürünlerin isimlerine bakınız.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 21987, 1)  # Out[48]: [21989]

arl_recommender(rules, 23235, 2) # Out[49]: [23244, 23243]

arl_recommender(rules, 22747, 3) # Out[50]: [22746, 22746, 22746]

""""
Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

"""

