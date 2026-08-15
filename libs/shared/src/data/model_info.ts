export interface DetailedModelSpec {
  id: string;
  name: string;
  category: 'Vision' | 'U-Net' | 'MediaPipe' | 'GenAI';
  weightsFile: string;
  architectureBackbone: string;
  inputTensor: string;
  quantization: string;
  downloadUrl: string;
  description: string;
  parametersExplanation: { [key: string]: string };
  mathOrFormula?: string;
}

export const MODEL_METADATA_REGISTRY: DetailedModelSpec[] = [
  {
    id: 'unet_custom',
    name: 'Configurable U-Net (PyTorch + CUDA)',
    category: 'U-Net',
    weightsFile: 'unet_custom_weights.pt',
    architectureBackbone: 'Encoder-Decoder with Skip Connections & Optional Attention Gates',
    inputTensor: '1 x 3 x 256 x 256 Float32',
    quantization: 'Float32 / Fused CUDA FP16 Acceleration',
    downloadUrl: '/api/unet/weights/export',
    description: 'A dynamic PyTorch U-Net architecture featuring customizable downsampling depth (2 to 5 levels), feature channel multipliers, Attention Gates, and fused CUDA activation kernels.',
    parametersExplanation: {
      'Network Depth': 'Controls the number of MaxPool2d downsampling stages in the encoder and ConvTranspose2d upsampling stages in the decoder.',
      'Initial Feature Channels': 'Number of convolution feature channels at level 0 (doubles at each deeper layer, e.g. 32 -> 64 -> 128 -> 256).',
      'Attention Gates': 'Applies spatial attention coefficients \\(\\alpha_i \\in [0, 1]\\) to skip connections to filter non-salient background noise.',
      'CUDA Fused Activation': 'Executes a custom C++/CUDA kernel combining Swish activation \\(\\text{SiLU}(x) = x \\cdot \\sigma(\\alpha x)\\) and spatial norm in a single GPU thread grid.'
    },
    mathOrFormula: '\\text{Attention Gate}: \\quad \\psi(g, x) = \\sigma\\left( W_z^T \\cdot \\delta(W_g^T g + W_x^T x + b_1) + b_2 \\right) \\cdot x'
  },
  {
    id: 'canny_edge',
    name: 'OpenCV Canny Edge Detector',
    category: 'Vision',
    weightsFile: 'N/A (Algorithmic Operator)',
    architectureBackbone: 'Multi-stage Edge Detection Algorithm (Gaussian Filter -> Sobel Gradient -> Non-Maximum Suppression -> Hysteresis Thresholding)',
    inputTensor: 'H x W Grayscale Image',
    quantization: 'N/A',
    downloadUrl: 'https://docs.opencv.org/4.x/da/222/tutorial_canny_technique.html',
    description: 'Classic multi-stage computer vision edge detector that computes spatial intensity gradients to detect thin, continuous structural contours.',
    parametersExplanation: {
      'Parameter 1 (Low Threshold)': 'Edges with intensity gradient value below this threshold are discarded as noise.',
      'Parameter 2 (High Threshold)': 'Edges with intensity gradient value above this threshold are marked as strong structural edges. Gradient values between Low and High are preserved if connected to strong edges.'
    },
    mathOrFormula: 'G = \\sqrt{G_x^2 + G_y^2}, \\quad \\theta = \\arctan\\left(\\frac{G_y}{G_x}\\right)'
  },
  {
    id: 'object_detector',
    name: 'MediaPipe Object Detector (EfficientDet-Lite0)',
    category: 'MediaPipe',
    weightsFile: 'efficientdet_lite0.tflite',
    architectureBackbone: 'EfficientNet-Lite0 Backbone + BiFPN Feature Pyramid Network',
    inputTensor: '320 x 320 x 3 RGB Normalized',
    quantization: 'Float16 Quantized TFLite',
    downloadUrl: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite',
    description: 'On-device object detection model trained on COCO dataset capable of identifying 80 common object classes with bounding box coordinates and confidence scores.',
    parametersExplanation: {
      'Score Threshold': 'Minimum confidence probability cutoff required to output object detection box.',
      'Max Results': 'Upper bound limit on total bounding boxes returned per image frame.'
    }
  },
  {
    id: 'selfie_segmenter',
    name: 'MediaPipe Selfie Segmenter',
    category: 'MediaPipe',
    weightsFile: 'selfie_segmenter.tflite',
    architectureBackbone: 'Custom MobileNetV3 Inverted Residuals Encoder',
    inputTensor: '256 x 256 x 3 RGB',
    quantization: 'Float16 Quantized',
    downloadUrl: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
    description: 'High-speed portrait segmentation model designed for separating human subjects (person foreground) from arbitrary backgrounds.',
    parametersExplanation: {
      'Mask Threshold': 'Probability threshold applied to output logit mask to generate binary foreground mask.'
    }
  },
  {
    id: 'face_landmarker',
    name: 'MediaPipe Face Landmarker',
    category: 'MediaPipe',
    weightsFile: 'face_landmarker.task',
    architectureBackbone: 'BlazeFace Short Range Detector + 3D Mesh Regressor',
    inputTensor: '192 x 192 x 3 RGB',
    quantization: 'Float16 Quantized',
    downloadUrl: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
    description: 'Estimates 468 3D facial landmarks in real-time, mapping facial contours, eyes, lips, and iris tracking points.',
    parametersExplanation: {
      'Num Faces': 'Maximum number of simultaneous faces tracked in camera stream.'
    }
  },
  {
    id: 'gemma_llm',
    name: 'MediaPipe LLM Inference (Gemma 2B)',
    category: 'GenAI',
    weightsFile: 'gemma-2b-it-cpu-int4.bin',
    architectureBackbone: 'Gemma 2B Transformer Decoder with Rotary Position Embeddings (RoPE)',
    inputTensor: 'Text Token Sequence',
    quantization: 'Int4 Quantized CPU Weights',
    downloadUrl: 'https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-2b-it-cpu-int4.bin',
    description: 'Lightweight, state-of-the-art open language model from Google optimized for on-device local text generation.',
    parametersExplanation: {
      'Temperature': 'Controls randomness in token sampling (lower value = deterministic, higher = creative).',
      'Max Tokens': 'Maximum token length limit for output generation.'
    }
  }
];
