import os
import json


# just to debug
print(os.path.basename(__file__), ' module name is: ', __name__)


def set_up_log_and_ws_out(models_out, opt_config, experiment_name, headers=None):
    target_logs = os.path.join(models_out, experiment_name + '/logs.csv')
    target_models = os.path.join(models_out, experiment_name)
    print('target_models', target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ';')

    if headers is None:
        log.writeHeaders(['epoch', 'train_loss', 'train_auc', 'train_map',
                          'val_loss', 'val_auc', 'val_map'])
    else:
        log.writeHeaders(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key] = value.__name__
            except:
                dump_cfg[key] = 'CrossEntropyLoss'
    json_cfg = os.path.join(models_out, experiment_name+'/cfg.json')
    with open(json_cfg, 'w') as json_file:
        json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out


class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')

def configure_backbone(backbone, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained=pretrained_arg, rgb_stack_size=size,
                    num_classes=num_classes_arg)
