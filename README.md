# VAE Latent Space Visualizer (Task 8)

Author: Ashif Raza

A Variational Autoencoder (VAE) trained on MNIST with a 2D latent space.
The Streamlit app shows the original image, the reconstructed image, and the latent space plot.

## Files
- `model.py` : VAE (encoder, latent sampling, decoder) and loss function
- `train.py` : trains the model and saves `vae.pth`
- `streamlit_app.py` : Streamlit web app
- `test_vae.py` : tests (architecture, latent distribution, reconstruction loss, KL divergence)

## How to run
```
pip install -r requirements.txt
python train.py      # skip if vae.pth already exists
streamlit run streamlit_app.py
```

## Run tests
```
python -m pytest test_vae.py -v
```

## Deploy
Push to GitHub, then create an app on https://share.streamlit.io with main file `streamlit_app.py`.
