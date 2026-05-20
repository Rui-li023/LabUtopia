import torch
import numpy as np
from typing import Dict
from loguru import logger

from .base_inference_engine import BaseInferenceEngine

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ModuleNotFoundError:
    logger.warning("OpenPI client not found. Please follow the instruction to install openpi-client")

class RemoteInferenceEngine(BaseInferenceEngine):
    """
    Remote inference engine using OpenPI client
    
    Connects to OpenPI server for remote inference using WebSocket communication
    """
    
    def _get_n_obs_steps(self) -> int:
        """Get observation steps from configuration"""
        return self.cfg.infer.n_obs_steps
    
    def _init_inference_engine(self):
        """Initialize OpenPI client connection"""
        # Get server connection parameters
        self.host = getattr(self.cfg.infer, 'host', '0.0.0.0')
        self.port = getattr(self.cfg.infer, 'port', None)
        self.api_key = getattr(self.cfg.infer, 'api_key', None)
        
        # Initialize OpenPI WebSocket client
        try:
            self.client = WebsocketClientPolicy(
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )
            
            # Get server metadata
            self.server_metadata = self.client.get_server_metadata()
            
            logger.success("OpenPI client initialized successfully")
            logger.info(f"Host: {self.host}, Port: {self.port}, Server metadata: {self.server_metadata}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenPI client: {e}")
            raise
    
    def _prepare_observation(self, obs_dict: Dict[str, torch.Tensor]) -> Dict:
        """
        Prepare observation data for OpenPI client
        
        Args:
            obs_dict: Dictionary containing observation tensors
            
        Returns:
            Dictionary formatted for OpenPI inference
        """
        observation = {}
        n_obs_steps = self._get_n_obs_steps()
        
        # Process each observation in the dictionary
        # Note: base_inference_engine only adds a batch dim when shape[0] != 1,
        # so for n_obs_steps==1 arrays are (1, ...) (time dim, no batch dim),
        # and for n_obs_steps>1 they are (1, n_obs_steps, ...).
        for obs_key, obs_tensor in obs_dict.items():
            arr = obs_tensor.cpu().numpy()
            if obs_key == 'agent_pose':
                # Server expects 'observation/state' as float32 [D] (1D), latest timestep
                latest_state = arr[0] if n_obs_steps == 1 else arr[0, -1]
                observation['observation/state'] = latest_state.astype(np.float32)
            else:
                if n_obs_steps == 1:
                    latest_image = arr[0]
                    # Camera may return CHW (3,H,W) or RGBA HWC (H,W,4).
                    # Normalize to HWC uint8 (H, W, 3).
                    if latest_image.ndim == 3 and latest_image.shape[0] in (1, 3, 4) and latest_image.shape[-1] not in (1, 3, 4):
                        latest_image = np.transpose(latest_image, (1, 2, 0))
                    if latest_image.dtype != np.uint8:
                        latest_image = (latest_image * 255).clip(0, 255).astype(np.uint8)
                    if latest_image.ndim == 3:
                        if latest_image.shape[2] > 3:
                            latest_image = latest_image[:, :, :3]
                        elif latest_image.shape[2] == 1:
                            latest_image = np.repeat(latest_image, 3, axis=2)
                    elif latest_image.ndim == 2:
                        latest_image = np.repeat(latest_image[:, :, np.newaxis], 3, axis=2)
                    # Training mp4 was stored as BGR (cv2 wrote BGR, training
                    # loader likely reads BGR without cvtColor). Swap to match.
                    if getattr(self.cfg.infer, "image_color", "bgr") == "bgr":
                        latest_image = latest_image[:, :, ::-1].copy()
                    observation[obs_key] = latest_image
                else:
                    images = arr[0]  # [time, C, H, W] or [time, H, W]
                    processed_images = []
                    for img in images:
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8)                        
                        processed_images.append(img)
                    observation[obs_key] = np.stack(processed_images, axis=0)
        
        return observation
    
    def _predict_action(self, obs_dict: Dict[str, torch.Tensor], language_instruction: str = "") -> np.ndarray:
        """
        Predict action using OpenPI client
        
        Args:
            obs_dict: Dictionary containing observation tensors
            
        Returns:
            Predicted action array
        """
        try:
            # Prepare observation data
            observation = self._prepare_observation(obs_dict)
            if not language_instruction:
                language_instruction = getattr(self.cfg.infer, 'prompt', '')
            observation['prompt'] = language_instruction

            self._infer_call_count = getattr(self, "_infer_call_count", 0) + 1
            if self._infer_call_count == 1:
                import cv2 as _cv2
                import os as _os
                _os.makedirs("/tmp/probe_live", exist_ok=True)
                for _k, _v in observation.items():
                    if isinstance(_v, np.ndarray) and _v.ndim == 3:
                        _name = _k.replace("/", "_")
                        # observation already in BGR (per fix), so write directly
                        _cv2.imwrite(f"/tmp/probe_live/{_name}.png", _v)
                logger.info("[debug] live obs dumped to /tmp/probe_live/")
            if self._infer_call_count <= 3:
                obs_summary = {}
                for k, v in observation.items():
                    if isinstance(v, np.ndarray):
                        obs_summary[k] = f"{v.dtype} {v.shape} min={float(v.min()):.4f} max={float(v.max()):.4f}"
                    else:
                        obs_summary[k] = repr(v)[:80]
                logger.info(f"[obs#{self._infer_call_count}] sending: {obs_summary}")

            result = self.client.infer(observation)

            if 'action' in result:
                action = np.array(result['action'])
            elif 'actions' in result:
                action = np.array(result['actions'])
            else:
                action_keys = [k for k in result.keys() if 'action' in k.lower()]
                if action_keys:
                    action = np.array(result[action_keys[0]])
                else:
                    raise ValueError(f"No action found in server response. Available keys: {list(result.keys())}")

            if self._infer_call_count <= 3:
                a_min = action.min(axis=0) if action.ndim > 1 else action
                a_max = action.max(axis=0) if action.ndim > 1 else action
                logger.info(
                    f"[act#{self._infer_call_count}] shape={action.shape} dtype={action.dtype} "
                    f"first={np.array2string(action[0] if action.ndim>1 else action, precision=3)} "
                    f"last={np.array2string(action[-1] if action.ndim>1 else action, precision=3)}"
                )
            return action
            
        except Exception as e:
            logger.error(f"OpenPI inference failed: {e}")
            # Return zero action as fallback
            return np.zeros((8, 8))  # Default action shape
    
    def close(self):
        """Close OpenPI client connection"""
        try:
            if hasattr(self, 'client'):
                self.client.reset()
            logger.success("OpenPI client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing OpenPI client: {e}")