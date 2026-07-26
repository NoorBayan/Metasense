import requests
import json
import pandas as pd
from json_repair import repair_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pyarabic.araby as araby

def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    return text

def load_and_prepare_data(url="https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/metaphors_data.json"):
    response = requests.get(url)
    fixed_json_string = repair_json(response.text)
    data = json.loads(fixed_json_string)

    records = []
    for item in data:
        ayah = item.get('metadata', {}).get('ayah_text_uthmani', '')
        # في البيانات الخاصة بك، الاستعارات توجد داخل قائمة similes
        metaphors = item.get('rhetorical_analysis', {}).get('similes', [])
        
        if not metaphors: continue
        
        for metaphor in metaphors:
            components = metaphor.get('components', {})
            identity = metaphor.get('simile_identity', {})
            
            sensory_mode = components.get('sensory_mode')
            segment_text = identity.get('segment_text', '')
            
            if sensory_mode and ayah:
                # دمج الآية مع مقطع الاستعارة للسياق المعرفي
                combined_text = f"{ayah} [SEP] {segment_text}" if segment_text else ayah
                records.append({'text': combined_text, 'label_text': sensory_mode})
                break 

    df = pd.DataFrame(records)
    # تنظيف البيانات من القيم الفارغة أو غير المتوقعة
    df = df.dropna(subset=['text', 'label_text'])
    df['clean_text'] = df['text'].apply(clean_arabic_text)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label_text'])

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
    
    return train_df, test_df, label_encoder
