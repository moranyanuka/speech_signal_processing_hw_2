#!/usr/bin/env python3
"""
DNS64 Inference Script for Colab
Run this in Google Colab to denoise signals and save outputs to denoiser_outputs folder.
"""

import torch
import torchaudio
from pathlib import Path
import glob
import os

# Install denoiser if not already installed
# !pip install denoiser

from denoiser import pretrained
from denoiser.dsp import convert_audio

def denoise_file(model, input_path, output_path, sr):
    """
    Load audio, denoise with DNS64, and save output.
    
    Args:
        model: DNS64 model
        input_path: path to noisy WAV file
        output_path: path to save denoised WAV file
        sr: original sample rate
    """
    try:
        # Load audio
        wav, file_sr = torchaudio.load(input_path)
        
        # Move to GPU if available
        device = next(model.parameters()).device
        wav = wav.to(device)
        
        # Convert to model sample rate if needed
        if file_sr != model.sample_rate:
            wav = torchaudio.transforms.Resample(file_sr, model.sample_rate)(wav)
        
        # Convert channels if needed (DNS64 expects mono)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Denoise
        with torch.no_grad():
            denoised = model(wav.unsqueeze(0))[0]  # Add batch dimension
        
        # Convert back to original sample rate if needed
        if model.sample_rate != sr:
            denoised = torchaudio.transforms.Resample(model.sample_rate, sr)(denoised)
        
        # Save output
        torchaudio.save(output_path, denoised.cpu(), sr)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading DNS64 model...")
    model = pretrained.dns64()
    model = model.to(device)
    model.eval()
    
    sr = 16000  # Sample rate (adjust if needed)
    
    # Input and output directories
    input_dir = './noisy_signals_for_denoising'
    output_dir = './denoiser_outputs'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all noisy WAV files
    noisy_files = sorted(glob.glob(f'{input_dir}/*.wav'))
    print(f"Found {len(noisy_files)} files to process\n")
    
    if len(noisy_files) == 0:
        print(f"No WAV files found in {input_dir}")
        return
    
    # Process all files
    successful = 0
    for idx, input_path in enumerate(noisy_files):
        fname = os.path.basename(input_path)
        output_path = os.path.join(output_dir, fname.replace('.wav', '_denoised.wav'))
        
        print(f"[{idx+1}/{len(noisy_files)}] Processing {fname}...", end=" ")
        
        if denoise_file(model, input_path, output_path, sr):
            print("✓")
            successful += 1
        else:
            print("✗")
    
    print(f"\n=== Complete ===")
    print(f"Successfully processed: {successful}/{len(noisy_files)}")
    print(f"Outputs saved to: {output_dir}/")

if __name__ == "__main__":
    main()
