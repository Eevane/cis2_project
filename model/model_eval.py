import torch
import numpy as np
import onnx
import onnxruntime
import os
from torch.utils.data import DataLoader
from model import JointLSTMModel, JointDataset
import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# input path
component = 'master2-Last'
path_root = "../../Dataset/0801_850Hz"
length = 5
testing_file_path = f"{path_root}/test/{component}ThreeJoints.csv"
model_path = f"{path_root}/checkpoints/len{length}/best-{component}.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_save = True
onnx_test = True
openvino_save = False

# load param
norm_data = np.load(f"{path_root}/checkpoints/len{length}/{component}-stat_params.npz")
input_mean = torch.tensor(norm_data['input_mean'], dtype=torch.float32)
input_std = torch.tensor(norm_data['input_std'], dtype=torch.float32)
target_mean = torch.tensor(norm_data['target_mean'], dtype=torch.float32)
target_std = torch.tensor(norm_data['target_std'], dtype=torch.float32)
seq_len = norm_data['seq_len']

model = JointLSTMModel()
pretrained_model = torch.load(model_path, map_location=device)
model.load_state_dict(pretrained_model['model_state_dict'])
model.to(device)
model.eval()

val_dataset = JointDataset(testing_file_path, seq_len=seq_len, mode='evaluation', in_mean=input_mean, in_std=input_std, tar_mean=target_mean, tar_std=target_std)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"test dataset length: {len(val_loader)}")

total_se = torch.zeros(3, device=device)
total_n = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)            
        pred = model(x)
        total_se += ((pred - y) ** 2).sum(dim=0)
        total_n += y.size(0)
mse_norm = total_se / total_n
rmse = torch.sqrt(mse_norm) * target_std.to(device)
nrmse_percent = rmse / val_dataset.target_range.to(device) * 100
print(f"NRMSE on validation set: {nrmse_percent.tolist()}%")
print("")

onnx_path = f"{path_root}/checkpoints/len{length}/"
if onnx_save:
    # output path
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)

    # save model into onnx format
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1,seq_len,6)

    torch.onnx.export(
        model,
        dummy_input,  # e.g. torch.randn(1, 3, 224, 224)
        onnx_path + f"{component}.onnx",
        export_params=True,
        input_names = ['input'],  
        output_names = ['output'],
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"}
        # }
        dynamic_axes=None
    )

    # verify if successfully save the model
    model_onnx = onnx.load(onnx_path + f"{component}.onnx")
    onnx.checker.check_model(model_onnx)

if onnx_test:
    # running under onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_path + f"{component}.onnx", providers=["CPUExecutionProvider"])

    start = time.time()
    for x, y in val_loader:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        ort_outs_denormal = ort_outs * to_numpy(target_std) + to_numpy(target_mean)
    print(f"running time is {time.time() - start}")
    print(f"ort_outs: {ort_outs}")
    # print(ort_outs_denormal)

if openvino_save:
    import openvino as ov
    ov_model = ov.convert_model(onnx_path + f"{component}.onnx")
    ov.save_model(ov_model, onnx_path + f"{component}.xml")

    # # test openvino runtime
    # compiled_model = ov.compile_model(ov_model, device_name="CPU")

    # start = time.time()
    # for x, y in val_loader:
    #     # result = compiled_model(to_numpy(x))[0]
    #     # result = list(compiled_model(to_numpy(x)).values())
    #     result = compiled_model(to_numpy(x))[0]
    #     openvino_result = result * to_numpy(target_std) + to_numpy(target_mean)
    # print(f"running time is {time.time() - start}")
    # print('')
    # print(f"result: {result}")
    # print(f"denormal result: {openvino_result}")