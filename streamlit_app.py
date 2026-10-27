# Project: VAE Latent Space Visualizer (Streamlit version)
# Author: Ashif Raza

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

from model import VAE

st.set_page_config(page_title="VAE Latent Space Visualizer", layout="wide")
st.title("VAE Latent Space Visualizer")
st.caption("A Variational Autoencoder trained on MNIST with a 2D latent space.")


@st.cache_resource
def load_model():
    model = VAE(latent_dim=2)
    model.load_state_dict(torch.load("vae.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_data():
    return datasets.MNIST("data", train=False, download=True,
                          transform=transforms.ToTensor())


@st.cache_data
def get_latent_points(n=2000):
    model, data = load_model(), load_data()
    xs = torch.stack([data[i][0] for i in range(n)])
    labels = np.array([int(data[i][1]) for i in range(n)])
    with torch.no_grad():
        mu, _ = model.encode(xs.view(-1, 784))
    return mu.numpy(), labels


def to_img(t):
    arr = (t.view(28, 28).detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((140, 140), Image.NEAREST)


model = load_model()
data = load_data()
points, labels = get_latent_points()

left, right = st.columns([1, 1])

with right:
    st.subheader("Latent space")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=6)
    fig.colorbar(sc, ax=ax, label="digit")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    plot_slot = st.empty()

with left:
    st.subheader("Reconstruction")
    uploaded = st.file_uploader("Upload a digit image (optional)",
                                type=["png", "jpg", "jpeg"])
    if st.button("Random test image") or "idx" not in st.session_state:
        st.session_state.idx = int(np.random.randint(len(data)))

    if uploaded is not None:
        img = Image.open(uploaded).convert("L").resize((28, 28))
        x = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    else:
        x = data[st.session_state.idx][0]

    with torch.no_grad():
        mu, _ = model.encode(x.view(-1, 784))
        recon = model.decode(mu)

    c1, c2 = st.columns(2)
    c1.image(to_img(x), caption="Original")
    c2.image(to_img(recon), caption="Reconstructed")
    z = mu[0].tolist()
    st.write(f"Latent position: z = ({z[0]:.2f}, {z[1]:.2f})")

# mark the current image's position on the latent plot
ax.scatter([z[0]], [z[1]], c="black", marker="X", s=150, edgecolors="white",
           label="current image")
ax.legend()
plot_slot.pyplot(fig)
