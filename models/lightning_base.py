from pytorch_lightning import LightningModule

class BaseLightningModel(LightningModule):
    
    def log_single_metric(self, metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=False):
        self.log(metric_name, metric_value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
    
    def log_single_dict(self, metric_dict, on_step=False, on_epoch=True, prog_bar=False):
        for k, v in metric_dict.items():
            self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)