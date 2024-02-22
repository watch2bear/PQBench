import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/fakenews/WELFake_Dataset.csv')
data=data.dropna()
X = data[['title','text']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

onnx_path='/fakenews/fakenews_nb.onnx'
ortconfig = ort.SessionOptions()
onnx_session1 = ort.InferenceSession(onnx_path, sess_options=ortconfig, providers=['AzureExecutionProvider', 'CPUExecutionProvider'] )
label1 = onnx_session1.get_outputs()[0]
inputs_info = onnx_session1.get_inputs()
input_names = [input_node.name for input_node in inputs_info]
input_columns = input_names
type_map = {
    'int32': np.int64,
    'int64': np.int64,
    'float64': np.float32,
    'object': str,
}
infer_batch = {
    elem: np.array(X_test[elem]).reshape((-1, 1)) for i, elem in enumerate(input_columns)
}
outputs = onnx_session1.run([label1.name], infer_batch)