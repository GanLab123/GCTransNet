
---

# GctransNet: 3D mitochondrial instance segmentation based on Global Context Vision Transformers.

GCTransNet is an automated mitochondria segmentation method based on 3D Global Context Vision Transformer (GC-ViT). By combining global and local self-attention modules, it effectively addresses the challenge of segmenting small mitochondrial targets in volume electron microscopy (EM) images. The method utilizes grayscale migration techniques for image preprocessing to reduce grayscale discrepancies between different slices, while precisely modeling both long-range and short-range spatial interactions. 

GCTransNet achieved state-of-the-art (SOTA) performance in the MitoEM Mitochondria Segmentation Challenge, significantly outperforming traditional methods.

## Abstract

Automated electron microscopy image segmentation techniques can enhance the efficiency and accuracy of mitochondrial research, offering new perspectives and possibilities for medical research. To address the challenge of segmenting small mitochondrial targets in volume electron microscopy images, we propose GCTransNet, an automated segmentation method for mitochondria. The method employs grayscale migration for image preprocessing, effectively reducing grayscale differences between slices. It utilizes 3D Global Context Vision Transformers combined with global and local self-attention modules to model long-range and short-range spatial interactions accurately. The encoder of the 3D U-Net architecture is replaced with the 3D GC-ViT, which leverages shifted windows for self-attention computation. In the MitoEM Mitochondria Segmentation Challenge, GCTransNet achieved SOTA results, significantly outperforming traditional methods, validating its superiority in automated mitochondria segmentation.

## Features

- **Grayscale Migration**: Reduces grayscale discrepancies across different slices, stabilizing segmentation results.
- **3D Global Context Vision Transformer (GC-ViT)**: Replaces the encoder in 3D U-Net with GC-ViT, using global and local self-attention to model spatial interactions accurately.
- **Efficient Mitochondria Segmentation**: Achieved state-of-the-art results in the MitoEM Challenge, outperforming traditional methods.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/GanLab123/GCTransNet.git
   cd GCTransNet
   ```

2. Create and activate a virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/MacOS
   .\venv\Scripts\activate   # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. train
   
```bash
python main.py
--data_dir
dataset/
--json_list
dataset.json
--logdir
GCTransNet
--out_channels
2
--roi_x
96
--roi_y
96
--roi_z
96
--a_min
0.0
--a_max
255.0
--max_epochs
2000
--batch_size
1
--save_checkpoint
```

## Contributing

Contributions are welcome! Please submit your pull requests to improve GCTransNet. To maintain code quality, ensure that your changes follow the project's coding standards and pass all tests.

## License

This project is licensed under the [MIT License](LICENSE).

---

This is a basic README template, and you may need to adjust or extend it based on your project specifics and needs.
