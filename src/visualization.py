import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def print_comparative_table(all_results):
    display_dict = {}
    for model, metrics in all_results.items():
        if model != 'TF-IDF (Baseline)':
            display_dict[model] = metrics
    
    df = pd.DataFrame(display_dict).T
    print("\n" + "="*80)
    print("🏆 Overall Performance Comparison for Sensory Mode (Mean ± Std)")
    print("="*80)
    print(df.to_markdown())
    print("="*80 + "\n")

def generate_reports(predictions_output, test_df, label_encoder, model_name):
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    target_names = label_encoder.classes_

    print(f"\n{'='*50}\n📊 Best Seed Classification Report: {model_name}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Sensory Mode Confusion Matrix ({model_name})", pad=15)
    plt.ylabel('Actual Sensory Mode')
    plt.xlabel('Predicted Sensory Mode')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{model_name}_cm.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def plot_radar_chart(model1_preds, model1_name, model2_preds, model2_name, label_encoder):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    y_true = model1_preds.label_ids
    y_pred1 = np.argmax(model1_preds.predictions, axis=1)
    y_pred2 = np.argmax(model2_preds.predictions, axis=1)
    
    classes = label_encoder.classes_
    
    # حساب F1-score لكل فئة للنموذجين
    f1_m1 = f1_score(y_true, y_pred1, average=None) * 100
    f1_m2 = f1_score(y_true, y_pred2, average=None) * 100
    
    # --- إضافة جديدة: طباعة القيم كجدول لتسهيل التحليل لكتابة الورقة ---
    print("\n" + "="*50)
    print("📊 F1-Scores per Sensory Mode (For Radar Chart Analysis)")
    print("="*50)
    df_radar = pd.DataFrame({
        'Sensory Mode': classes,
        f'{model1_name} (F1 %)': np.round(f1_m1, 2),
        f'{model2_name} (F1 %)': np.round(f1_m2, 2)
    })
    print(df_radar.to_markdown(index=False))
    print("="*50 + "\n")
    # ------------------------------------------------------------------

    # تجهيز بيانات الرسم
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    f1_m1_plot = np.concatenate((f1_m1, [f1_m1[0]]))
    f1_m2_plot = np.concatenate((f1_m2, [f1_m2[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # رسم النموذج الأول
    ax.fill(angles, f1_m1_plot, color='blue', alpha=0.1)
    ax.plot(angles, f1_m1_plot, color='blue', linewidth=2, label=model1_name)
    
    # رسم النموذج الثاني
    ax.fill(angles, f1_m2_plot, color='red', alpha=0.1)
    ax.plot(angles, f1_m2_plot, color='red', linewidth=2, linestyle='dashed', label=model2_name)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_title("Cognitive Sensory Modes: F1-Score Comparison", size=15, pad=20)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('radar_chart.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def run_mcnemar_test(model1_preds, model2_preds, model1_name, model2_name):
    y_true = model1_preds.label_ids
    y_pred1 = np.argmax(model1_preds.predictions, axis=1)
    y_pred2 = np.argmax(model2_preds.predictions, axis=1)
    
    both_correct = sum((y_pred1 == y_true) & (y_pred2 == y_true))
    m1_only = sum((y_pred1 == y_true) & (y_pred2 != y_true))
    m2_only = sum((y_pred1 != y_true) & (y_pred2 == y_true))
    both_wrong = sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    contingency = [[both_correct, m1_only], [m2_only, both_wrong]]
    result = mcnemar(contingency, exact=True)
    
    print("\n" + "="*60)
    print("🔬 Statistical Significance (McNemar's Test)")
    print("="*60)
    print(f"- {model1_name} got {m1_only} right that {model2_name} got wrong.")
    print(f"- {model2_name} got {m2_only} right that {model1_name} got wrong.")
    print(f"\n=> p-value: {result.pvalue:.5f}")
    
    if result.pvalue < 0.05:
        print(f"✅ The difference between models IS statistically significant!")
    else:
        print(f"❌ The difference is NOT statistically significant.")
    print("="*60 + "\n")
