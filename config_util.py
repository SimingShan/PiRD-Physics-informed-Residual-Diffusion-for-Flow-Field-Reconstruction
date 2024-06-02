from pydantic import BaseModel
class DatasetConfig(BaseModel):
    path: str
    transform: str
    image_size: int

class CorruptionConfig(BaseModel):
    method: str
    scale: int
    portion: float

class VisualizationConfig(BaseModel):
    sample_index: int
    channel: int

class DiffusionConfig(BaseModel):
    target: str
    sf: int
    schedule_name: str
    etas_end: float
    steps: int
    min_noise_level: float
    kappa: float
    weighted_mse: bool
    predict_type: str
    scale_factor: float
    normalize_input: bool
    latent_flag: bool
    kwargs: float
    num_diffusion_steps: int

class ModelConfig(BaseModel):
    type: str
    in_channels: int
    out_ch: int
    ch: int
    ch_mult: list
    num_res_blocks: int
    attn_resolutions: tuple
    dropout: float
    var_type: str
    ema_rate: float
    ema: bool
    resamp_with_conv: bool
    ckpt_path: str

class TrainConfig(BaseModel):
    epoch: int
    lr: float
    MSE_weight: float
    ADV_weight: float
    DIF_weight: float
    BC_weight: float
    save_interval: int
    batch_size: int


class AppConfig(BaseModel):
    dataset: DatasetConfig
    corruption: CorruptionConfig
    visualization: VisualizationConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    train: TrainConfig

