import re

with open('app.py', 'r') as f:
    content = f.read()

content = re.sub(r'gender = st\.selectbox\("Gender \(0=Female, 1=Male\)", options=sorted\(\[int\(x\) for x in ranges\[\'Gender\'\]\[\'unique\'\]\]\)\)\s*', '', content)
content = re.sub(r"'Gender': gender,\s*", '', content)
content = re.sub(r"'Gender': \(\"The dataset patterns show a correlation \(likely synthetic\)\.\", \"Maintain overall health \(this is likely a dataset artifact\)\.\"\),\s*", '', content)
content = content.replace('Age, Gender, Genetic Risk', 'Age, Genetic Risk')
content = content.replace('st.markdown("### Patient Data Input")', 'st.markdown("### Patient Data Input")\n    st.info("Note: \'Gender\' was excluded from this model because it was identified as a non-causal dataset artifact.")')

with open('app.py', 'w') as f:
    f.write(content)
