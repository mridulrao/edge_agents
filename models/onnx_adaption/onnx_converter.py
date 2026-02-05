#!/usr/bin/env python3
"""
ONNX Model Converter
Converts Hugging Face instruction-tuned models to ONNX format with various quantizations.

Usage:
    python onnx_converter.py --model HuggingFaceTB/SmolLM2-360M-Instruct --quantize int8
    python onnx_converter.py --model microsoft/phi-2 --quantize fp16
    python onnx_converter.py --model HuggingFaceTB/SmolLM2-360M-Instruct --list-formats
"""

import argparse
import shutil
from pathlib import Path
from typing import Literal, Optional
import logging

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, QuantizationConfig
from transformers import AutoTokenizer, AutoConfig
import onnx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXConverter:
    """Convert HuggingFace models to ONNX with quantization options"""
    
    SUPPORTED_QUANTIZATIONS = {
        'fp32': 'Full precision (largest, slowest)',
        'fp16': 'Half precision (good balance)',
        'int8': '8-bit quantization (recommended for edge)',
        'int4': '4-bit quantization (smallest, experimental)',
    }
    
    def __init__(self, output_base_dir: str = "./onnx_models"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
    
    def convert(
        self,
        model_id: str,
        quantization: Literal['fp32', 'fp16', 'int8', 'int4'] = 'int8',
        output_name: Optional[str] = None,
        optimize: bool = True,
    ) -> Path:
        """
        Convert a HuggingFace model to ONNX with specified quantization.
        
        Args:
            model_id: HuggingFace model ID (e.g., "HuggingFaceTB/SmolLM2-360M-Instruct")
            quantization: Target quantization format
            output_name: Custom output directory name (default: auto-generated)
            optimize: Apply ONNX graph optimizations
            
        Returns:
            Path to the converted model directory
        """
        logger.info(f"Converting {model_id} to ONNX ({quantization})")
        
        # Generate output directory name
        if output_name is None:
            model_name = model_id.split('/')[-1].lower().replace('-', '_')
            output_name = f"{model_name}_onnx_{quantization}"
        
        output_dir = self.output_base_dir / output_name
        temp_dir = self.output_base_dir / f"temp_{output_name}"
        
        try:
            # Step 1: Convert to ONNX (FP32 base)
            logger.info("Exporting to ONNX format...")
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                provider="CPUExecutionProvider",
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            config = AutoConfig.from_pretrained(model_id)
            
            # Save to temp directory
            ort_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            config.save_pretrained(temp_dir)
            
            logger.info("Base ONNX export complete")
            
            # Step 2: Apply quantization if needed
            if quantization == 'fp32':
                # Just move temp to output
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                shutil.move(str(temp_dir), str(output_dir))
                logger.info(f"FP32 model saved to {output_dir}")
            
            elif quantization == 'fp16':
                self._convert_to_fp16(temp_dir, output_dir)
                shutil.rmtree(temp_dir)
            
            elif quantization == 'int8':
                self._quantize_to_int8(temp_dir, output_dir)
                shutil.rmtree(temp_dir)
            
            elif quantization == 'int4':
                self._quantize_to_int4(temp_dir, output_dir)
                shutil.rmtree(temp_dir)
            
            # Step 3: Report statistics
            self._print_stats(output_dir, model_id, quantization)
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def _convert_to_fp16(self, input_dir: Path, output_dir: Path):
        """Convert ONNX model to FP16"""
        logger.info("Converting to FP16...")
        
        from onnxconverter_common import float16
        
        # Find ONNX model file
        onnx_files = list(input_dir.glob("*.onnx"))
        if not onnx_files:
            onnx_files = list(input_dir.glob("**/*.onnx"))
        
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {input_dir}")
        
        model_path = onnx_files[0]
        
        # Load and convert
        model = onnx.load(str(model_path))
        model_fp16 = float16.convert_float_to_float16(model)
        
        # Save
        output_dir.mkdir(exist_ok=True, parents=True)
        output_model_path = output_dir / model_path.name
        onnx.save(model_fp16, str(output_model_path))
        
        # Copy tokenizer and config
        for file in input_dir.glob("*"):
            if file.suffix in ['.json', '.txt', '.model']:
                shutil.copy(file, output_dir / file.name)
        
        logger.info(f"FP16 model saved to {output_dir}")
    
    def _quantize_to_int8(self, input_dir: Path, output_dir: Path):
        """Quantize ONNX model to INT8"""
        logger.info("Quantizing to INT8...")
        
        quantizer = ORTQuantizer.from_pretrained(str(input_dir))
        
        # Dynamic quantization config (no calibration data needed)
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=True
        )
        
        quantizer.quantize(
            save_dir=str(output_dir),
            quantization_config=qconfig
        )
        
        logger.info(f"INT8 model saved to {output_dir}")
    
    def _quantize_to_int4(self, input_dir: Path, output_dir: Path):
        """Quantize ONNX model to INT4 (experimental)"""
        logger.info("Quantizing to INT4...")
        
        quantizer = ORTQuantizer.from_pretrained(str(input_dir))
        
        # INT4 quantization
        from optimum.onnxruntime.configuration import QuantizationMode
        
        qconfig = QuantizationConfig(
            is_static=False,
            format=QuantizationMode.QInt4,
            per_channel=True,
        )
        
        quantizer.quantize(
            save_dir=str(output_dir),
            quantization_config=qconfig
        )
        
        logger.info(f"INT4 model saved to {output_dir}")
    
    def _print_stats(self, model_dir: Path, model_id: str, quantization: str):
        """Print conversion statistics"""
        # Find ONNX file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            onnx_files = list(model_dir.glob("**/*.onnx"))
        
        if onnx_files:
            size_mb = onnx_files[0].stat().st_size / (1024 * 1024)
            
            logger.info("=" * 60)
            logger.info("📊 Conversion Summary")
            logger.info("=" * 60)
            logger.info(f"Model ID:       {model_id}")
            logger.info(f"Quantization:   {quantization}")
            logger.info(f"Output Path:    {model_dir}")
            logger.info(f"Model Size:     {size_mb:.1f} MB")
            logger.info(f"ONNX File:      {onnx_files[0].name}")
            logger.info("=" * 60)
    
    @classmethod
    def list_formats(cls):
        """List supported quantization formats"""
        print("\nSupported Quantization Formats:\n")
        for fmt, desc in cls.SUPPORTED_QUANTIZATIONS.items():
            print(f"  {fmt:8} - {desc}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to ONNX with quantization"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model ID (e.g., HuggingFaceTB/SmolLM2-360M-Instruct)"
    )
    
    parser.add_argument(
        "--quantize",
        type=str,
        choices=['fp32', 'fp16', 'int8', 'int4'],
        default='int8',
        help="Quantization format (default: int8)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_models",
        help="Base output directory (default: ./onnx_models)"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom output directory name (default: auto-generated)"
    )
    
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported quantization formats and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_formats:
        ONNXConverter.list_formats()
        return
    
    if not args.model:
        parser.error("--model is required (use --list-formats to see options)")
    
    # Convert
    converter = ONNXConverter(output_base_dir=args.output_dir)
    
    try:
        output_path = converter.convert(
            model_id=args.model,
            quantization=args.quantize,
            output_name=args.output_name,
        )
        
        print(f"\nSuccess! Model saved to: {output_path}\n")
        
    except Exception as e:
        print(f"\nError: {e}\n")
        exit(1)


if __name__ == "__main__":
    main()