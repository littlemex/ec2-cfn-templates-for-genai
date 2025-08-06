#!/usr/bin/env python3
"""
JAX環境情報をJSON形式で生成するスクリプト
"""
import json
import sys
import subprocess
import os

def get_jax_environment_info():
    """JAX環境の詳細情報を取得してJSON形式で返す"""
    info = {}
    
    # 環境パス
    info["environment_path"] = os.environ.get("VIRTUAL_ENV", "")
    
    # Python情報
    try:
        python_version = subprocess.check_output([sys.executable, "--version"], 
                                               stderr=subprocess.STDOUT).decode().strip()
        info["python_version"] = python_version
    except:
        info["python_version"] = "unknown"
    
    info["python_path"] = sys.executable
    
    # JAX情報
    jax_info = {"available": False}
    try:
        import jax
        import jax.numpy as jnp
        jax_info["available"] = True
        jax_info["version"] = jax.__version__
        
        # JAXlib バージョン
        try:
            import jaxlib
            jax_info["jaxlib_version"] = jaxlib.__version__
        except:
            jax_info["jaxlib_version"] = "unknown"
        
        # デバイス情報
        try:
            devices = jax.devices()
            device_count = len(devices)
            neuron_count = len([d for d in devices if 'neuron' in str(d).lower()])
            jax_info["devices"] = f"Found {device_count} devices ({neuron_count} Neuron devices)"
            jax_info["device_count"] = device_count
            jax_info["neuron_device_count"] = neuron_count
        except:
            jax_info["devices"] = "Device query failed"
            jax_info["device_count"] = 0
            jax_info["neuron_device_count"] = 0
        
        # デフォルトバックエンド
        try:
            jax_info["default_backend"] = jax.default_backend()
        except:
            jax_info["default_backend"] = "unknown"
            
    except ImportError:
        jax_info["version"] = "not_installed"
        jax_info["jaxlib_version"] = "not_installed"
        jax_info["devices"] = "JAX not available"
        jax_info["default_backend"] = "unknown"
    
    info["jax"] = jax_info
    
    # JAX-NeuronX情報
    jax_neuronx_info = {"available": False}
    try:
        import jax_neuronx
        jax_neuronx_info["available"] = True
        try:
            jax_neuronx_info["version"] = jax_neuronx.__version__
        except:
            jax_neuronx_info["version"] = "unknown"
    except ImportError:
        jax_neuronx_info["version"] = "not_installed"
    
    info["jax_neuronx"] = jax_neuronx_info
    
    # libneuronxla情報
    libneuronxla_info = {"available": False}
    try:
        import libneuronxla
        libneuronxla_info["available"] = True
        try:
            libneuronxla_info["version"] = libneuronxla.__version__
        except:
            libneuronxla_info["version"] = "unknown"
        
        # サポートされているクライアント
        try:
            supported_clients = libneuronxla.supported_clients()
            libneuronxla_info["supported_clients"] = str(supported_clients)
        except:
            libneuronxla_info["supported_clients"] = "N/A"
    except ImportError:
        libneuronxla_info["version"] = "not_installed"
        libneuronxla_info["supported_clients"] = "N/A"
    
    info["libneuronxla"] = libneuronxla_info
    
    # neuronx-cc情報
    neuronx_cc_info = {"available": False}
    try:
        import neuronxcc
        neuronx_cc_info["available"] = True
        try:
            neuronx_cc_info["version"] = neuronxcc.__version__
        except:
            neuronx_cc_info["version"] = "unknown"
    except ImportError:
        neuronx_cc_info["version"] = "not_installed"
    
    info["neuronx_cc"] = neuronx_cc_info
    
    # インストール済みパッケージ（Neuron関連）
    try:
        pip_output = subprocess.check_output([sys.executable, "-m", "pip", "list"], 
                                           stderr=subprocess.DEVNULL).decode()
        neuron_packages = []
        for line in pip_output.split('\n'):
            if 'neuron' in line.lower():
                neuron_packages.append(line.strip())
        info["neuron_packages"] = '\n'.join(neuron_packages) if neuron_packages else "No neuron packages found"
    except:
        info["neuron_packages"] = "Package list query failed"
    
    return info

if __name__ == "__main__":
    try:
        jax_info = get_jax_environment_info()
        print(json.dumps(jax_info, indent=2, ensure_ascii=False))
    except Exception as e:
        error_info = {"error": str(e), "available": False}
        print(json.dumps(error_info, indent=2, ensure_ascii=False))
        sys.exit(1)