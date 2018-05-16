import torch
from modules.models.inception import Inception3


class ModelHandler:
    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

    @staticmethod
    def get_new_model(gpu_mode):
        model = Inception3()
        if gpu_mode:
            model = model.cuda()
        return model

    @staticmethod
    def load_model(model_path, gpu_mode):
        model = torch.load(model_path)
        if gpu_mode is True:
            model = model.cuda()
        return model

    @staticmethod
    def load_optimizer(optimizer, checkpoint_path, gpu_mode):
        if gpu_mode:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])

        return optimizer

    @staticmethod
    def load_model_for_training(state_dict_path, gpu_mode):
        if gpu_mode:
            model = ModelHandler.load_gpu_models_to_cpu(state_dict_path)
            model = model.cuda()
        else:
            model = ModelHandler.load_gpu_models_to_cpu(state_dict_path)

        return model

    @staticmethod
    def load_gpu_models_to_cpu(state_dict_path):
        checkpoint = torch.load(state_dict_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model = Inception3()
        model.load_state_dict(new_state_dict)
        model.cpu()

        return model
