import onnx
from onnx import compose


class ModelMerger:
    def __init__(self,model_paths, output_path):
        self.model_paths = model_paths
        self.output_path = output_path
        self.model_names = ['master1_l_first','master1_l_last','master1_r_first','master1_r_last'
                            ,'master2_l_first','master2_l_last','master2_r_first','master2_r_last',
                            'puppet1_first','puppet1_last','puppet2_first','puppet2_last']

    def merge_models(self):
        """
        Merges multiple ONNX models into a single model.
        The sequence should be ['master1_first','master1_last','master2_first','master2_last','puppet_first','puppet_last']

        Args:
            model_paths (list of str): List of paths to the ONNX model files to be merged.
            output_path (str): Path where the merged ONNX model will be saved.

        Returns: 
            None
        """
        models = []
        for i,model_path in enumerate(self.model_paths):
            model = onnx.load(model_path)
            model = compose.add_prefix(model, self.model_names[i])
            models.append(model)
        
        # merge the models
        merged_model = models[0]
        for model in models[1:]:
            merged_model = compose.merge_models(merged_model, model, io_map=[])

        # Save the merged model
        onnx.save(merged_model, self.output_path)

        # check
        print(f"Merged model with {len(models)} sub-models to {self.output_path}")


if __name__ == "__main__": 
    pass

