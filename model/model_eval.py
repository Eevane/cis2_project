import torch
import numpy as np
import onnx
import onnxruntime
from torch.utils.data import DataLoader
from model_seq import JointLSTMModel, JointDataset
#from model import JointDataset

if __name__ == "__main__":
    component = 'puppet-Last'
    testing_file_path = f"../../Dataset/test_0627/{component}ThreeJoints.csv"
    model_path = f"../../Dataset/checkpoints/0628-{component}.pth.tar"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onnx_save = False

    # load param 
    norm_data = np.load(f"../../Dataset/checkpoints/0628-{component}-norm_params.npz")
    input_mean = norm_data['input_mean']
    input_std = norm_data['input_std']
    target_mean = norm_data['target_mean']
    target_std = norm_data['target_std']

    model = JointLSTMModel()
    pretrained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_model['model_state_dict'])
    model.to(device)
    model.eval()

    val_dataset = JointDataset(testing_file_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    total_se = torch.zeros(3, device=device)
    total_n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)            
            pred = model(x)
            total_se += ((pred - y) ** 2).sum(dim=0)
            total_n += y.size(0)
    mse = total_se / total_n
    rmse = torch.sqrt(mse)
    nrmse_percent = rmse * val_dataset.target_std.to(device) / val_dataset.target_range.to(device) * 100
    print(f"NRMSE on validation set: {nrmse_percent.tolist()}%")
    print("")

    if onnx_save:
        # save model into onnx format
        model.eval()
        model.cpu()
        dummy_input = torch.randn(1,1,6)

        torch.onnx.export(
            model,
            dummy_input,  # e.g. torch.randn(1, 3, 224, 224)
            f"../../Dataset/checkpoints/onnx/0628-Mul-{component}.onnx",
            export_params=True,
            input_names = ['input'],  
            output_names = ['output'],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )


        # verify if successfully save the model
        model_onnx = onnx.load(f"../../Dataset/checkpoints/0628-Mul-{component}.onnx")
        onnx.checker.check_model(model_onnx)

        # running under onnxruntime
        ort_session = onnxruntime.InferenceSession(f"../../Dataset/checkpoints/0628-Mul-{component}.onnx", providers=["CPUExecutionProvider"])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        for x, y in val_loader:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
            ort_outs = ort_session.run(None, ort_inputs)
            ort_outs_denormal = ort_outs * target_std + target_mean
            # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print(ort_outs)
        print(ort_outs_denormal)



