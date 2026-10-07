import re
with open('src/explain_one.py', 'r') as f:
    content = f.read()

# Add calibrator loading in the functions
def_fc = """
def factor_contributions(raw_row_df, model, encoder, baseline=None, data_path='dataset/cancer_data.csv'):
    calibrator = None
    try:
        calibrator = load_calibrator('models/calibrator.pkl')
    except:
        pass
        
    def pred_cal(encoded):
        p = float(model.predict(encoded, verbose=0)[0][0])
        if calibrator:
            p = float(calibrator.predict([p])[0])
        return min(p, 0.999)
"""
content = re.sub(r'def factor_contributions\(.*?\):', def_fc.strip(), content)
content = content.replace('p_full = float(model.predict(encoded_full, verbose=0)[0][0])', 'p_full = pred_cal(encoded_full)')
content = content.replace('p_base = float(model.predict(encoded_base, verbose=0)[0][0])', 'p_base = pred_cal(encoded_base)')
content = content.replace('p_replaced = float(model.predict(encoded_replaced, verbose=0)[0][0])', 'p_replaced = pred_cal(encoded_replaced)')

def_wi = """
def what_if(raw_row_df, model, encoder, quartiles=None, data_path='dataset/cancer_data.csv'):
    calibrator = None
    try:
        calibrator = load_calibrator('models/calibrator.pkl')
    except:
        pass
        
    def pred_cal(encoded):
        p = float(model.predict(encoded, verbose=0)[0][0])
        if calibrator:
            p = float(calibrator.predict([p])[0])
        return min(p, 0.999)
"""
content = re.sub(r'def what_if\(.*?\):', def_wi.strip(), content)
content = content.replace('p_full = float(model.predict(encode(raw_row_df, encoder), verbose=0)[0][0])', 'p_full = pred_cal(encode(raw_row_df, encoder))')
content = content.replace('p_new = float(model.predict(encode(row, encoder), verbose=0)[0][0])', 'p_new = pred_cal(encode(row, encoder))')

with open('src/explain_one.py', 'w') as f:
    f.write(content)
