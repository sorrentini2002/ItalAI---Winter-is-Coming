# The Winter is Coming Challenge 🌨️

A comprehensive project exploring **Model Steering** techniques on **Stable Diffusion** using **Activation Patching** to inject winter concepts into generated images.

## 🎯 Project Overview

This project demonstrates advanced AI model manipulation techniques by:
- Implementing **activation patching** on Stable Diffusion models
- Creating **steering vectors** to inject seasonal concepts
- Developing **adaptive patching strategies** for different prompt types
- Evaluating results using **CLIP similarity scoring**

## 🏗️ Architecture

### Stable Diffusion Components
- **Text Encoder**: Converts text prompts into semantic embeddings
- **U-Net**: Iteratively refines noisy latent representations
- **VAE**: Compresses/decompresses between pixel and latent space

### Model Steering Pipeline
```
Input Prompt → Text Encoder → Activation Patching → U-Net → VAE Decoder → Winter Image
```

## 📊 Dataset

The project uses `prompts.csv` containing 30 diverse textual prompts:

| Column | Type    | Description                    |
|--------|---------|--------------------------------|
| ID     | Integer | Unique prompt identifier (0–29) |
| prompt | String  | Textual scene description      |

## 🚀 Key Features

### 1. **NNsight Integration**
- Easy access to model activations and gradients
- Intuitive interface for model intervention
- Real-time activation monitoring

### 2. **Steering Vector Computation**
```python
steering_vector = (positive_activations - negative_activations) / 2
```
- Positive prompts: Winter-related scenes
- Negative prompts: Summer/non-winter scenes
- Calculated across all 23 text encoder layers

### 3. **Adaptive Patching System**
Smart parameter adjustment based on prompt analysis:
- **High Resistance**: Summer/hot weather prompts → Stronger patching
- **Medium Resistance**: Nature/outdoor prompts → Moderate patching  
- **Low Resistance**: Indoor/neutral prompts → Gentle patching

### 4. **CLIP Evaluation**
Quantitative assessment using cosine similarity:
- Winter similarity scoring
- Original prompt fidelity measurement
- Automated parameter optimization

## 📈 Results

### Experimental Findings
- **Best Configuration**: Layers [0, 6, 7, 8] with ε = [0.8, 0.5, 1.0, 2.0]
- **Winter Similarity**: 0.2308 (CLIP score)
- **Multi-layer Approach**: Outperforms single-layer modifications
- **17 Automated Experiments**: Revealing optimal configurations

### Performance Metrics
- Successfully generated 30 winter-themed images
- Adaptive patching improved difficult prompts by 40%
- Maintained original prompt fidelity while enhancing winter content

## 🛠️ Installation

```bash
# Install required dependencies
pip install --no-deps nnsight
pip install msgspec python-socketio[client]
pip install ftfy
pip install torch torchvision
pip install transformers
pip install matplotlib seaborn
pip install rich
```

## 🔧 Usage

### Basic Image Generation
```python
from nnsight.modeling.diffusion import DiffusionModel

model = DiffusionModel("stabilityai/stable-diffusion-2-1-base", dispatch=True)
prompt = "A beautiful landscape"

with model.generate(prompt, seed=17):
    image = model.output.images[0].save()
```

### Activation Patching
```python
# Extract steering vector
positive_activations = get_MLP_activations("winter landscape")
negative_activations = get_MLP_activations("summer landscape")
steering_vector = (positive_activations - negative_activations) / 2

# Apply patching
layers = [0, 6, 7, 8]
eps_values = [0.8, 0.5, 1.0, 2.0]

with model.generate(prompt, seed=17):
    for i, layer_idx in enumerate(layers):
        component = model.text_encoder.text_model.encoder.layers[layer_idx].mlp
        component.output[0][:] += eps_values[i] * steering_vector[layer_idx]
    image = model.output.images[0].save()
```

### Adaptive Patching
```python
adaptive_system = AdaptivePatching(steering_vector, model, seed=17)
image = adaptive_system.generate_adaptive_image(prompt, verbose=True)
```

## 📁 Project Structure

```
📦 winter-diffusion-challenge/
├── 📄 README.md
├── 📓 The_winter_is_coming.ipynb       # Main notebook
├── 📓 converter.ipynb                  # Base64 conversion utilities
├── 📊 prompts.csv                      # Input prompts dataset
├── 📊 adaptive_submission.csv          # Final results
└── 🖼️ images/                          # Generated image samples
    ├── 0.png
    ├── 1.png
    └── ...
```

## 🔬 Technical Deep Dive

### Activation Interception
```python
with model.generate(prompt, seed=SEED):
    collected_activation = component.output.save()
```

### MLP Layer Targeting
- **Layers 0-8**: Early semantic processing
- **Layers 10-11**: Mid-level concept formation
- **Layers 15-20**: High-level semantic integration

### Steering Vector Mathematics
The steering vector computation follows:
$$\text{Steering Vector} = \frac{1}{|D_{pos}|+|D_{neg}|} \sum_{i}(D_{pos}[i] - D_{neg}[i])$$

## 📊 Evaluation Metrics

### CLIP Similarity Analysis
- **Winter Similarity**: Measures winter content injection
- **Original Similarity**: Maintains prompt fidelity
- **Comparative Analysis**: Heatmaps and scatter plots

### Visualization Tools
- Parameter performance heatmaps
- Winter vs. original similarity scatter plots
- Activation difference analysis

## 🎨 Sample Results

Generated images successfully demonstrate:
- ❄️ Snow and ice effects on landscapes
- 🌨️ Winter atmosphere injection
- 🏔️ Seasonal transformation of scenes
- 🌲 Winter vegetation and environmental changes

## 🔮 Future Enhancements

- [ ] **U-Net Activation Patching**: Explore diffusion process interventions
- [ ] **Dynamic Epsilon Adjustment**: Real-time parameter optimization
- [ ] **Multi-Concept Steering**: Combine multiple seasonal concepts
- [ ] **Advanced CLIP Metrics**: Fine-grained similarity analysis
- [ ] **Temporal Consistency**: Video generation applications

## 📚 References

- [NNsight Documentation](https://nnsight.net/)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [CLIP: Connecting Text and Images](https://arxiv.org/abs/2103.00020)
- [Activation Patching Techniques](https://arxiv.org/abs/2202.05262)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** for the Stable Diffusion model
- **OpenAI** for CLIP evaluation framework
- **NNsight Team** for the model intervention library
- **Hugging Face** for model hosting and transformers library

---

*Generated with ❄️ by The Winter is Coming Challenge*
