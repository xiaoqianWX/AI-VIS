"""AI-VIS module"""
import gc
import os
import numpy as np
import torch
import torchvision.transforms as transform
import skimage.transform as sktransform
from safetensors.torch import load_file

class AI_VIS:
    def __init__(self, gpu_id='0'):
        """Init AI-VIS module.

        Args:
            gpu_id (str): Set GPU device to use. If set -1, will disable GPU device enforcely.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpu_id is None else gpu_id
        self.gpu_id = gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load(self, weight_path='./aivis/weights', upscale=False, half_precision=False, tile=0, tile_pad=10, pre_pad=0):
        """Load AI-VIS model, including its module and weights.

        Args:
            weight_path (str): The path to the pretrained model.
            upscale (bool, optional): Whether to use super-resolution (Real-ESRGAN-x4-plus) model.
            half_precision (bool, optional): Whether to use half precision (fp16) on super-resolution model.
            tile (int, optional): Crop input images into tiles for saving GPU memory. See `models.realesrgan.RealESRGANer` for more information.
            tile_pad (int, optional): The pad size for each tile, to remove border artifacts.
            pre_pad (int, optional): Pad the input images to avoid border artifacts.
        """  
        self.upscale = upscale
        
        # load AI-VIS model weight and checkpoint
        from .models.aivis_1_0 import GeneratorUNet
        self._Gpath = os.path.join(weight_path, 'aivis_1_0.safetensors')
        self.G = GeneratorUNet().to(self.device)
        self._ckpt = load_file(self._Gpath)
        self.G.load_state_dict(self._ckpt)
        self.G.eval()
            
        self.T = transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.T_inverse = transform.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
        if self.upscale:
            # load upscaling model (Real-ESRGAN-x4-plus)
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from .models.realesrgan import RealESRGANer
            self._Upath = os.path.join(weight_path, 'upscaler_1_5.safetensors')
            state_dict = load_file(os.path.join(weight_path, 'upscaler_1_5.safetensors'), device='cpu')
            self._Umodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._Umodel.load_state_dict(state_dict)
            self.U = RealESRGANer(
                model_path=self._Upath,
                model=self._Umodel,
                scale=4, dni_weight=None,
                tile=tile, tile_pad=tile_pad, pre_pad=pre_pad,
                half=half_precision, gpu_id=self.gpu_id
            )
            self.U.model = self._Umodel
            # use bilinear interpolation with skimage module
            # only use on lon/lat interpolation
            self.interpolate = lambda lonlat: (
                # x4 upscale from (500, 500) to (2000, 2000)
                sktransform.rescale(lonlat, 4, order=1)
            )
    
    def release(self):
        """Release GPU memory / RAM usage.
        """
        del self.G, self._Gpath, self._ckpt, self.T
        
        if self.upscale:
            del self.U, self._Upath, self._Umodel
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def data_upscale(self, lons, lats, data):
        """Make a super-resolution to AI-VIS output data, and interpolate longitude and latitude to the same size.

        Args:
            lons (np.ndarray): Input longitude data to interpolate.
            lats (np.ndarray): Input latitude data to interpolate.
            data (np.ndarray): Input AI-VIS output data to Real-ESRGANx4 model.
        """
        if not lons.shape == lats.shape == data.shape == (500, 500):
            raise ValueError("Size must be (500, 500)")
        
        lons = self.interpolate(lons)
        lats = self.interpolate(lats)
        
        with torch.no_grad():
            data = np.stack((data.copy(), data.copy(), data.copy()), axis=2)
            data, _ = self.U.enhance(data * 255, outscale=4)
            data = data[:, :, 0] / 255
        
        return lons, lats, data
    
    def data_to_aivis(self, lons, lats, datas, basemap, sza, az, sat_za, sat_az):
        """Use AI-VIS model that inputs Infrared data to simulate visible (BAND 3).
        For input data, usually uses 'B08', 'B09', 'B10', 'B11', 'B13', 'B15', 'B16' on HIM-8/9 in extract order.

        Args:
            lons (np.ndarray): Input longitude data. Will use if need upscaling.
            lats (np.ndarray): Input latitude data. Will use if need upscaling.
            datas (np.ndarray): Input infrared data to AI-VIS model.
            basemap (np.ndarray): Input basemap to AI-VIS model.
            sza (float): Input sun zenith angle to AI-VIS model.
            az (float): Input sun azimuth angle to AI-VIS model.
            sat_za (float): Input satellite zenith angle to AI-VIS model.
            sat_az (float): Input satellite azimuth angle to AI-VIS model.
        
        """
        if not lons.shape == lats.shape == (500, 500):
            raise ValueError("Size of lon/lat array must be (500, 500)")
        
        if not datas.shape == (7, 500, 500):
            raise ValueError("Size of data must be (7, 500, 500)")
        
        if sza is None or az is None:
            raise ValueError("`data_to_aivis` function must give `sza` and `az` a float")
        
        if sat_za is None or sat_az is None:
            raise ValueError("`data_to_aivis` function must give `sat_za` and `sat_az` a float")
        
        # extract data in the exact order
        bt08, bt09, bt10, bt11, bt13, bt15, bt16 = datas
        
        # process data
        bt13_08 = bt13 - bt08
        bt13_09 = bt13 - bt09
        bt13_10 = bt13 - bt10
        bt11_15 = bt11 - bt15
        bt13_15 = bt13 - bt15
        bt13_16 = bt13 - bt16
        
        bt13_10 = 1 - np.clip(bt13_10 + 12, 0, 74) / 74
        bt11_15 = 1 - np.clip(bt11_15 + 12, 0, 34) / 34
        bt13_16 = 1 - np.clip(bt13_16 + 3, 0, 44) / 44
        bt13_08 = 1 - np.clip(bt13_08 + 11, 0, 91) / 91
        bt13_09 = 1 - np.clip(bt13_09 + 10, 0, 80) / 80
        bt13_15 = 1 - np.clip(bt13_15 + 3, 0, 25) / 25
        bt13 = 1 - np.clip(bt13 + 103, 0, 148) / 148
        basemap = basemap / 255
        
        side1 = np.ones([6, 500, 3])
        side2 = np.ones([512, 6, 3])
        
        sza_color = np.zeros([500, 500]) + 1 - np.clip(sza, 0, 90) / 90
        az_color = np.zeros([500, 500]) + (np.clip(az, -180, 180) + 180) / 360
        
        sat_az_color = np.zeros([500, 500]) + np.clip(sat_az, 0, 360) / 360
        sat_za_color = np.zeros([500, 500]) + np.clip(sat_za, 0, 90) / 90
        
        Band13_15_az = np.stack([bt13, bt13_15, az_color], axis=2)
        Band13_15_az = np.concatenate([side1, Band13_15_az, side1], axis=0)
        Band13_15_az = np.concatenate([side2, Band13_15_az, side2], axis=1)
        Band13_15_az = np.uint8(np.clip(Band13_15_az * 255, 0., 255.))
        
        Band10_11_16 = np.stack([bt13_10, bt11_15, bt13_16], axis=2)
        Band10_11_16 = np.concatenate([side1, Band10_11_16, side1], axis=0)
        Band10_11_16 = np.concatenate([side2, Band10_11_16, side2], axis=1)
        Band10_11_16 = np.uint8(np.clip(Band10_11_16 * 255, 0., 255.))
        
        Band08_09_sza = np.stack([bt13_08, bt13_09, sza_color], axis=2)
        Band08_09_sza = np.concatenate([side1, Band08_09_sza, side1], axis=0)
        Band08_09_sza = np.concatenate([side2, Band08_09_sza, side2], axis=1)
        Band08_09_sza = np.uint8(np.clip(Band08_09_sza * 255, 0., 255.))
        
        Band_sataz_satza_map = np.stack([sat_az_color, sat_za_color, basemap], axis=2)
        Band_sataz_satza_map = np.concatenate([side1, Band_sataz_satza_map, side1], axis=0)
        Band_sataz_satza_map = np.concatenate([side2, Band_sataz_satza_map, side2], axis=1)
        Band_sataz_satza_map = np.uint8(np.clip(Band_sataz_satza_map * 255, 0., 255.))
        
        # transform numpy array to torch tensor
        img = torch.cat([self.T(Band13_15_az), self.T(Band10_11_16), self.T(Band08_09_sza), self.T(Band_sataz_satza_map)], dim=0)
        img = img[None].to(self.device)
        
        with torch.no_grad():
            out = self.T_inverse(self.G(img)[0])
            out = out.detach().permute(1, 2, 0).cpu().numpy()
            out = out[6:-6, 6:-6, 0] # delete the white-side
        
        if self.upscale:
            lons, lats, out = self.data_upscale(lons, lats, out)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return lons, lats, out
