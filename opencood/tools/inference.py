# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import torchvision
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.seg_iou import mean_IU
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

# ============ DEBUG CONFIGURATION ============
DEBUG_SEGMENTATION = False  # 🔧 Set to False to disable all debug output
# =============================================

def debug_print(message, level="INFO"):
    """Enhanced debug print with clear formatting"""
    if DEBUG_SEGMENTATION:
        if level == "ERROR":
            print(f"\n🔴 [SEG_DEBUG-{level}] {message}")
        elif level == "WARNING":
            print(f"🟡 [SEG_DEBUG-{level}] {message}")
        elif level == "SUCCESS":
            print(f"🟢 [SEG_DEBUG-{level}] {message}")
        else:
            print(f"🔵 [SEG_DEBUG-{level}] {message}")

def debug_tensor_info(tensor, name, detailed=True):
    """Print detailed tensor information for debugging"""
    if not DEBUG_SEGMENTATION:
        return
        
    print(f"\n{'='*50}")
    print(f"🔍 [SEG_DEBUG] TENSOR ANALYSIS: {name}")
    print(f"{'='*50}")
    
    if tensor is None:
        debug_print(f"{name} is None!", "ERROR")
        return
    
    try:
        if torch.is_tensor(tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = np.array(tensor)
            
        print(f"📐 Shape: {tensor_np.shape}")
        print(f"📊 Data type: {tensor_np.dtype}")
        print(f"📈 Value range: [{tensor_np.min():.6f}, {tensor_np.max():.6f}]")
        print(f"🎯 Unique values: {np.unique(tensor_np)}")
        print(f"📝 Total elements: {tensor_np.size}")
        
        if detailed:
            # Count pixels for each class
            unique_vals, counts = np.unique(tensor_np, return_counts=True)
            print(f"📊 Class distribution:")
            for val, count in zip(unique_vals, counts):
                percentage = (count / tensor_np.size) * 100
                print(f"   Class {int(val)}: {count:8d} pixels ({percentage:6.2f}%)")
                
        print(f"{'='*50}\n")
        
    except Exception as e:
        debug_print(f"Error analyzing tensor {name}: {e}", "ERROR")

def calculate_segmentation_metrics(pred, gt, num_classes, class_names=None):
    """
    Calculate comprehensive segmentation metrics including IoU, Precision, Recall, F1, Accuracy
    
    Parameters
    ----------
    pred : numpy.ndarray
        Predicted segmentation map [H, W]
    gt : numpy.ndarray  
        Ground truth segmentation map [H, W]
    num_classes : int
        Number of classes
    class_names : list, optional
        Names of classes for better reporting
        
    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Flatten arrays for sklearn metrics
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Overall pixel accuracy
    overall_accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        gt_flat, pred_flat, labels=range(num_classes), zero_division=0
    )
    
    # IoU calculation
    iou_per_class = []
    for i in range(num_classes):
        intersection = np.sum((pred_flat == i) & (gt_flat == i))
        union = np.sum((pred_flat == i) | (gt_flat == i))
        if union == 0:
            iou_per_class.append(0.0 if intersection == 0 else 1.0)
        else:
            iou_per_class.append(intersection / union)
    
    # Mean IoU
    mean_iou = np.mean(iou_per_class)
    
    # Frequency weighted IoU
    freq_weights = support / np.sum(support)
    freq_weighted_iou = np.sum(freq_weights * iou_per_class)
    
    # Class accuracy (recall)
    class_accuracy = recall
    
    # Mean class accuracy
    mean_class_accuracy = np.mean(class_accuracy)
    
    return {
        'overall_accuracy': overall_accuracy,
        'mean_iou': mean_iou,
        'freq_weighted_iou': freq_weighted_iou,
        'mean_class_accuracy': mean_class_accuracy,
        'per_class_metrics': {
            class_names[i]: {
                'iou': iou_per_class[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            } for i in range(num_classes)
        },
        'confusion_matrix': conf_matrix
    }

def print_segmentation_metrics(metrics, title="Segmentation Metrics"):
    """
    Print segmentation metrics in a formatted way
    """
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"🎯 Overall Pixel Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"🔥 Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"⚖️  Frequency Weighted IoU: {metrics['freq_weighted_iou']:.4f}")
    print(f"📈 Mean Class Accuracy: {metrics['mean_class_accuracy']:.4f}")
    
    # Per-class metrics
    print(f"\n{'='*40}")
    print(f"📋 Per-Class Metrics:")
    print(f"{'='*40}")
    print(f"{'Class':<15} {'IoU':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Support':<8}")
    print(f"{'-'*65}")
    
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"{class_name:<15} "
              f"{class_metrics['iou']:<8.4f} "
              f"{class_metrics['precision']:<10.4f} " 
              f"{class_metrics['recall']:<8.4f} "
              f"{class_metrics['f1']:<8.4f} "
              f"{class_metrics['support']:<8}")
    
    print(f"{'='*60}\n")

def create_segmentation_metrics_summary():
    """
    Create a summary of all available segmentation metrics
    """
    summary = """
    🎯 ENHANCED SEGMENTATION EVALUATION METRICS
    ==========================================
    
    📊 CLASSIC METRICS (similar to AP in detection):
    ├─ IoU (Intersection over Union) - Per class and mean
    ├─ Pixel Accuracy - Overall correctness
    ├─ Mean Class Accuracy - Balanced accuracy across classes
    └─ Frequency Weighted IoU - Weighted by class frequency
    
    🔍 DETAILED PER-CLASS METRICS:
    ├─ Precision - True positives / (True positives + False positives)
    ├─ Recall - True positives / (True positives + False negatives)  
    ├─ F1-Score - Harmonic mean of precision and recall
    └─ Support - Number of true instances per class
    
    🚗 VEHICLE DETECTION SPECIFIC:
    ├─ Vehicle IoU - How well vehicles are segmented
    ├─ Vehicle F1-Score - Balanced vehicle detection performance
    ├─ Vehicle Precision - Accuracy of vehicle predictions
    └─ Vehicle Recall - Completeness of vehicle detection
    
    📈 VISUALIZATION & ANALYSIS:
    ├─ Confusion Matrix - Class-wise prediction analysis
    ├─ Class Distribution - Pixel counts per class
    ├─ Per-frame detailed metrics saved to JSON
    └─ Comprehensive final results summary
    
    🔧 USAGE:
    ├─ Add --detailed_seg_metrics for comprehensive metrics
    ├─ Add --seg_debug for detailed debugging information
    └─ Metrics saved to segmentation_results_[info].json
    
    ==========================================
    """
    return summary

def eval_segmentation_result(opt, infer_result_single, i, work_dir):
    """
    Calculate IoU for segmentation task during inference.
    
    Parameters
    ----------
    opt : argparse.Namespace
        Command line arguments
    infer_result_single : dict
        Single inference result containing predictions and ground truth
    i : int
        Current iteration index
    work_dir : str
        Working directory for saving results
        
    Returns
    -------
    tuple
        IoU for static and dynamic segmentation
    """
    debug_print(f"Starting segmentation evaluation for frame {i}", "INFO")
    
    pred_dict = infer_result_single["pred_box_tensor"]
    gt_dict = infer_result_single["gt_box_tensor"]
    
    if pred_dict is None or gt_dict is None:
        debug_print("pred_dict or gt_dict is None!", "ERROR")
        return None, None
    
    debug_print("Checking input dictionaries structure...")
    if DEBUG_SEGMENTATION:
        print(f"🔍 [SEG_DEBUG] pred_dict keys: {list(pred_dict.keys())}")
        print(f"🔍 [SEG_DEBUG] gt_dict keys: {list(gt_dict.keys())}")
    
    # === GROUND TRUTH ANALYSIS ===
    debug_print("Analyzing ground truth data...", "INFO")
    
    batch_size = gt_dict["static_bev"].shape[0]
    debug_print(f"Batch size: {batch_size}")
    assert batch_size == 1, "Only support batch size 1 for now."

    gt_static = gt_dict["static_bev"].detach().cpu().data.numpy()[0]
    gt_static = np.array(gt_static, dtype=int)
    debug_tensor_info(gt_static, "GT_STATIC", detailed=True)

    gt_dynamic = gt_dict["dynamic_bev"].detach().cpu().data.numpy()[0]
    gt_dynamic = np.array(gt_dynamic, dtype=int)
    debug_tensor_info(gt_dynamic, "GT_DYNAMIC", detailed=True)

    # === PREDICTION ANALYSIS ===
    debug_print("Analyzing prediction data...", "INFO")
    
    # Raw predictions before processing
    pred_static_raw = pred_dict["static_map"]
    pred_dynamic_raw = pred_dict["dynamic_map"]
    
    debug_tensor_info(pred_static_raw, "PRED_STATIC_RAW", detailed=True)
    debug_tensor_info(pred_dynamic_raw, "PRED_DYNAMIC_RAW", detailed=True)

    # Apply center crop and convert to numpy
    debug_print("Applying center crop and type conversion...", "INFO")
    
    pred_static = torchvision.transforms.CenterCrop(gt_static.shape)(pred_static_raw[0]).detach().cpu().data.numpy()
    pred_static = np.array(pred_static, dtype=int)
    debug_tensor_info(pred_static, "PRED_STATIC_PROCESSED", detailed=True)

    pred_dynamic = torchvision.transforms.CenterCrop(gt_dynamic.shape)(pred_dynamic_raw[0]).detach().cpu().data.numpy()
    pred_dynamic = np.array(pred_dynamic, dtype=int)
    debug_tensor_info(pred_dynamic, "PRED_DYNAMIC_PROCESSED", detailed=True)
    
    # === COMPARISON ANALYSIS ===
    debug_print("Comparing predictions vs ground truth...", "INFO")
    
    if DEBUG_SEGMENTATION:
        # Static comparison
        static_correct = np.sum(pred_static == gt_static)
        static_total = gt_static.size
        static_accuracy = static_correct / static_total * 100
        print(f"🔍 [SEG_DEBUG] Static pixel accuracy: {static_accuracy:.2f}% ({static_correct}/{static_total})")
        
        # Dynamic comparison
        dynamic_correct = np.sum(pred_dynamic == gt_dynamic)
        dynamic_total = gt_dynamic.size
        dynamic_accuracy = dynamic_correct / dynamic_total * 100
        print(f"🔍 [SEG_DEBUG] Dynamic pixel accuracy: {dynamic_accuracy:.2f}% ({dynamic_correct}/{dynamic_total})")
        
        # Check if model predicts any vehicles
        gt_vehicles = np.sum(gt_dynamic == 1)
        pred_vehicles = np.sum(pred_dynamic == 1)
        print(f"🔍 [SEG_DEBUG] Vehicle pixels - GT: {gt_vehicles}, Predicted: {pred_vehicles}")
        
        if gt_vehicles > 0 and pred_vehicles == 0:
            debug_print("⚠️  MODEL FAILED TO PREDICT ANY VEHICLES!", "ERROR")
        elif pred_vehicles > 0:
            debug_print(f"✅ Model predicted {pred_vehicles} vehicle pixels", "SUCCESS")
    
    # === IoU CALCULATION & COMPREHENSIVE METRICS ===
    debug_print("Calculating comprehensive segmentation metrics...", "INFO")
    
    # Define class names for better reporting
    static_class_names = ["Background", "Road", "Lane"]
    dynamic_class_names = ["Background", "Vehicle"]
    
    # Calculate comprehensive metrics
    static_metrics = None
    dynamic_metrics = None
    
    try:
        # Static segmentation metrics
        static_metrics = calculate_segmentation_metrics(
            pred_static, gt_static, 
            num_classes=3, 
            class_names=static_class_names
        )
        debug_print(f"Static Mean IoU: {static_metrics['mean_iou']:.4f}", "SUCCESS")
        if DEBUG_SEGMENTATION:
            print_segmentation_metrics(static_metrics, "Static Segmentation Metrics")
            
        # Legacy IoU for compatibility
        iou_static = mean_IU(pred_static, gt_static)
        debug_tensor_info(iou_static, "STATIC_IoU", detailed=False)
        
    except Exception as e:
        debug_print(f"Error calculating static metrics: {e}", "ERROR")
        iou_static = None
    
    try:
        # Dynamic segmentation metrics  
        dynamic_metrics = calculate_segmentation_metrics(
            pred_dynamic, gt_dynamic,
            num_classes=2,
            class_names=dynamic_class_names
        )
        debug_print(f"Dynamic Mean IoU: {dynamic_metrics['mean_iou']:.4f}", "SUCCESS")
        if DEBUG_SEGMENTATION:
            print_segmentation_metrics(dynamic_metrics, "Dynamic Segmentation Metrics")
            
        # Legacy IoU for compatibility
        iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
        debug_tensor_info(iou_dynamic, "DYNAMIC_IoU", detailed=False)
        
    except Exception as e:
        debug_print(f"Error calculating dynamic metrics: {e}", "ERROR")
        iou_dynamic = None
    
    # Print summary metrics
    if static_metrics and dynamic_metrics:
        print(f"\n🎯 SUMMARY METRICS - Frame {i}:")
        print(f"   Static  - Mean IoU: {static_metrics['mean_iou']:.4f}, Pixel Acc: {static_metrics['overall_accuracy']:.4f}")
        print(f"   Dynamic - Mean IoU: {dynamic_metrics['mean_iou']:.4f}, Pixel Acc: {dynamic_metrics['overall_accuracy']:.4f}")
        
        # Vehicle detection performance
        if 'Vehicle' in static_metrics['per_class_metrics']:
            vehicle_f1 = dynamic_metrics['per_class_metrics']['Vehicle']['f1']
            vehicle_recall = dynamic_metrics['per_class_metrics']['Vehicle']['recall']
            vehicle_precision = dynamic_metrics['per_class_metrics']['Vehicle']['precision']
            print(f"   Vehicle Detection - F1: {vehicle_f1:.4f}, Recall: {vehicle_recall:.4f}, Precision: {vehicle_precision:.4f}")
        print()
        
    # Save comprehensive metrics to file
    if i % opt.save_vis_interval == 0 and (static_metrics or dynamic_metrics):
        metrics_save_path = os.path.join(work_dir, 'detailed_metrics')
        if not os.path.exists(metrics_save_path):
            os.makedirs(metrics_save_path)
            
        metrics_file = os.path.join(metrics_save_path, f"metrics_frame_{i:05d}.json")
        
        combined_metrics = {
            'frame_id': i,
            'static_metrics': static_metrics,
            'dynamic_metrics': dynamic_metrics
        }
        
        import json
        with open(metrics_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2, default=str)
        
        debug_print(f"Detailed metrics saved to {metrics_file}", "SUCCESS")

    # === VISUALIZATION ===
    if i % opt.save_vis_interval == 0:
        debug_print("Generating visualization...", "INFO")
        
        vis_save_path_root = os.path.join(work_dir, f'vis_segmentation')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)

        save_path = os.path.join(vis_save_path_root, "%05d_bev_seg.png" % i)
        static_save_path = os.path.join(vis_save_path_root, "%05d_bev_static.png" % i)
        dynamic_save_path = os.path.join(vis_save_path_root, "%05d_bev_dynamic.png" % i)

        static_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_static.png" % i)
        dynamic_gt_save_path = os.path.join(vis_save_path_root, "%05d_gt_dynamic.png" % i)

        colors = [(255, 255, 255), (255, 200, 200), (20, 20, 220), (80, 40, 40)]
        seg_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        static_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dynamic_gt = np.ones((256, 256, 3), dtype=np.uint8) * 255

        for j in range(3):
            seg_image[pred_static == j] = colors[j]
            static_image[pred_static == j] = colors[j]
            static_gt[gt_static == j] = colors[j]
        seg_image[pred_dynamic == 1] = colors[3]
        dynamic_image[pred_dynamic == 1] = colors[3]
        dynamic_gt[gt_dynamic == 1] = colors[3]
        
        cv2.imwrite(save_path, seg_image)
        cv2.imwrite(static_save_path, static_image)
        cv2.imwrite(dynamic_save_path, dynamic_image)
        cv2.imwrite(static_gt_save_path, static_gt)
        cv2.imwrite(dynamic_gt_save_path, dynamic_gt)
        
        debug_print(f"Visualizations saved to {vis_save_path_root}", "SUCCESS")

    debug_print(f"Segmentation evaluation completed for frame {i}", "SUCCESS")
    
    # Return both legacy IoU and comprehensive metrics
    return_dict = {
        'iou_static': iou_static,
        'iou_dynamic': iou_dynamic, 
        'static_metrics': static_metrics,
        'dynamic_metrics': dynamic_metrics
    }
    return return_dict

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--all', action='store_true', help="evaluate all the agents instead of the first one.")
    parser.add_argument('--show_bev', action='store_true', help="Visualize the BEV feature")
    parser.add_argument('--detailed_seg_metrics', action='store_true', 
                        help="Calculate and display detailed segmentation metrics (IoU, Precision, Recall, F1)")
    parser.add_argument('--seg_debug', action='store_true', 
                        help="Enable debug mode for segmentation evaluation")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    
    # Set debug mode based on command line argument
    global DEBUG_SEGMENTATION
    if opt.seg_debug:
        DEBUG_SEGMENTATION = True
    
    # Debug status notification
    if DEBUG_SEGMENTATION:
        print("\n" + "="*60)
        print("🔧 SEGMENTATION DEBUG MODE ENABLED 🔧")
        print("   Detailed debug information will be displayed")
        print("   To disable: remove --seg_debug flag")
        print("="*60 + "\n")
    
    # Show available metrics if detailed metrics are enabled
    if opt.detailed_seg_metrics:
        print(create_segmentation_metrics_summary())

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"


        # Handle different postprocess structures
        if 'postprocess' in hypes:
            # Direct postprocess structure
            if 'anchor_args' in hypes['postprocess']:
                new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                                    x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
                # replace all appearance
                hypes = update_dict(hypes, {
                    "cav_lidar_range": new_cav_range,
                    "lidar_range": new_cav_range,
                    "gt_range": new_cav_range
                })
            else:
                # To handle heter task case, where different modality has different postprocessor
                for modality_name, p in hypes['postprocess'].items():
                    assert modality_name[0] == 'm' and modality_name[1:].isdigit()
                    if 'anchor_args' in p:
                        new_cav_range = [x_min, y_min, p['anchor_args']['cav_lidar_range'][2], \
                                        x_max, y_max, p['anchor_args']['cav_lidar_range'][5]]
                    update_dict(p, {
                        "cav_lidar_range": new_cav_range,
                        "lidar_range": new_cav_range,
                        "gt_range": new_cav_range
                    })
        elif 'heter' in hypes and 'modality_setting' in hypes['heter']:
            # Nested postprocess structure in heter.modality_setting
            for modality_name, modality_config in hypes['heter']['modality_setting'].items():
                if 'postprocess' in modality_config and 'anchor_args' in modality_config['postprocess']:
                    new_cav_range = [x_min, y_min, modality_config['postprocess']['anchor_args']['cav_lidar_range'][2], \
                                    x_max, y_max, modality_config['postprocess']['anchor_args']['cav_lidar_range'][5]]
                    update_dict(modality_config['postprocess'], {
                        "cav_lidar_range": new_cav_range,
                        "lidar_range": new_cav_range,
                        "gt_range": new_cav_range
                    })
                # Also update top-level ranges
                hypes = update_dict(hypes, {
                    "cav_lidar_range": new_cav_range,
                    "lidar_range": new_cav_range,
                    "gt_range": new_cav_range
                })


        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        
        # Handle both yaml_parser (singular) and yaml_parsers (plural) cases
        if "yaml_parser" in hypes:
            parser_name = hypes["yaml_parser"]
            if hasattr(yaml_utils_lib, parser_name):
                parser_func = getattr(yaml_utils_lib, parser_name)
                hypes = parser_func(hypes)
        elif "yaml_parsers" in hypes:
            # For heterogeneous case with multiple parsers
            for modality_name, parser_name in hypes["yaml_parsers"].items():
                if hasattr(yaml_utils_lib, parser_name):
                    parser_func = getattr(yaml_utils_lib, parser_name)
                    if 'modality_setting' in hypes['heter'] and modality_name in hypes['heter']['modality_setting']:
                        hypes['heter']['modality_setting'][modality_name] = parser_func(hypes['heter']['modality_setting'][modality_name])
        
        # Legacy code for backward compatibility
        # for name, func in yaml_utils_lib.__dict__.items():
        #     if name == hypes.get("yaml_parser", ""):
        #         parser_func = func
        #         hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    # Set default postprocess range if not present (for segmentation tasks)
    if 'postprocess' not in hypes:
        # For heterogeneous segmentation tasks, extract postprocess from modality settings
        if 'heter' in hypes and 'modality_setting' in hypes['heter']:
            ego_modality = hypes['heter'].get('ego_modality', 'm1')
            if ego_modality in hypes['heter']['modality_setting'] and 'postprocess' in hypes['heter']['modality_setting'][ego_modality]:
                # Copy the postprocess config to top level for dataset building
                hypes['postprocess'] = hypes['heter']['modality_setting'][ego_modality]['postprocess'].copy()
            else:
                # Create a minimal postprocess config for segmentation tasks
                default_range = hypes.get('cav_lidar_range', [-51.2, -51.2, -2, 51.2, 51.2, 2])
                hypes['postprocess'] = {
                    'core_method': 'CameraBevPostprocessor',  # Default for segmentation
                    'gt_range': default_range,
                    'anchor_args': {
                        'cav_lidar_range': default_range
                    },
                    'max_num': 100,
                    'nms_thresh': 0.15,
                    'order': 'hwl'
                }
        else:
            # Use cav_lidar_range as default for visualization
            default_range = hypes.get('cav_lidar_range', [-51.2, -51.2, -2, 51.2, 51.2, 2])
            hypes['postprocess'] = {
                'core_method': 'CameraBevPostprocessor',
                'gt_range': default_range
            }
    elif 'gt_range' not in hypes['postprocess']:
        # Fallback to cav_lidar_range or model range
        if 'cav_lidar_range' in hypes:
            hypes['postprocess']['gt_range'] = hypes['cav_lidar_range']
        elif 'heter' in hypes and 'modality_setting' in hypes['heter']:
            # Get range from first modality
            first_modality = list(hypes['heter']['modality_setting'].keys())[0]
            if 'postprocess' in hypes['heter']['modality_setting'][first_modality]:
                hypes['postprocess']['gt_range'] = hypes['heter']['modality_setting'][first_modality]['postprocess'].get('gt_range', [-51.2, -51.2, -2, 51.2, 51.2, 2])
            else:
                hypes['postprocess']['gt_range'] = [-51.2, -51.2, -2, 51.2, 51.2, 2]
        else:
            hypes['postprocess']['gt_range'] = [-51.2, -51.2, -2, 51.2, 51.2, 2]

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    modality_list = opencood_dataset.modality_name_list
    # Create the dictionary for evaluation
    if opt.all:
        result_stat = {m: {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}} for m in modality_list}
        # Add segmentation stats
        seg_result_stat = {m: {'static_iou': [], 'dynamic_iou': []} for m in modality_list}
    else:
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        # Add segmentation stats
        seg_result_stat = {'static_iou': [], 'dynamic_iou': []}

    
    infer_info = opt.fusion_method + opt.note


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                infer_all=opt.all,
                                                                show_bev=opt.show_bev)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            
            agent_modality_list = batch_data['ego']['agent_modality_list']
            if not opt.all:
                infer_result = [infer_result]
            for idx, infer_result_single in enumerate(infer_result):
                if opt.all:
                    work_dir = os.path.join(opt.model_dir, f'modality_{agent_modality_list[idx]}')
                    os.makedirs(work_dir, exist_ok=True)
                else:
                    work_dir = opt.model_dir
                
                # Check if this is a segmentation task result (dictionary format)
                if isinstance(infer_result_single.get('pred_box_tensor'), dict):
                    # Segmentation task - run segmentation evaluation
                    debug_print(f"Detected segmentation task on modality {agent_modality_list[idx] if opt.all else 'default'}", "INFO")
                    
                    eval_result = eval_segmentation_result(opt, infer_result_single, i, work_dir)
                    
                    if eval_result and eval_result['iou_static'] is not None and eval_result['iou_dynamic'] is not None:
                        # Extract legacy IoU values for compatibility
                        iou_static = eval_result['iou_static']
                        iou_dynamic = eval_result['iou_dynamic']
                        static_metrics = eval_result['static_metrics']
                        dynamic_metrics = eval_result['dynamic_metrics']
                        
                        if opt.all:
                            seg_result_stat[agent_modality_list[idx]]['static_iou'].append(iou_static)
                            seg_result_stat[agent_modality_list[idx]]['dynamic_iou'].append(iou_dynamic)
                            # Store comprehensive metrics too
                            if 'static_metrics' not in seg_result_stat[agent_modality_list[idx]]:
                                seg_result_stat[agent_modality_list[idx]]['static_metrics'] = []
                                seg_result_stat[agent_modality_list[idx]]['dynamic_metrics'] = []
                            seg_result_stat[agent_modality_list[idx]]['static_metrics'].append(static_metrics)
                            seg_result_stat[agent_modality_list[idx]]['dynamic_metrics'].append(dynamic_metrics)
                        else:
                            seg_result_stat['static_iou'].append(iou_static)
                            seg_result_stat['dynamic_iou'].append(iou_dynamic)
                            # Store comprehensive metrics too
                            if 'static_metrics' not in seg_result_stat:
                                seg_result_stat['static_metrics'] = []
                                seg_result_stat['dynamic_metrics'] = []
                            seg_result_stat['static_metrics'].append(static_metrics)
                            seg_result_stat['dynamic_metrics'].append(dynamic_metrics)
                        
                        # Legacy display for compatibility
                        static_mean = np.mean(iou_static)
                        dynamic_mean = np.mean(iou_dynamic)
                        
                        # Enhanced display with comprehensive metrics
                        if static_metrics and dynamic_metrics:
                            static_overall_acc = static_metrics['overall_accuracy']
                            dynamic_overall_acc = dynamic_metrics['overall_accuracy']
                            static_mean_iou = static_metrics['mean_iou']
                            dynamic_mean_iou = dynamic_metrics['mean_iou']
                            
                            print(f"Frame {i}: Static IoU: {static_mean_iou:.4f} (Acc: {static_overall_acc:.4f}), "
                                  f"Dynamic IoU: {dynamic_mean_iou:.4f} (Acc: {dynamic_overall_acc:.4f})")
                            
                            debug_print(f"Frame {i} Comprehensive Results - "
                                      f"Static [IoU: {static_mean_iou:.4f}, Acc: {static_overall_acc:.4f}], "
                                      f"Dynamic [IoU: {dynamic_mean_iou:.4f}, Acc: {dynamic_overall_acc:.4f}]", "SUCCESS")
                        else:
                            print(f"Frame {i}: Static IoU: {static_mean:.4f}, Dynamic IoU: {dynamic_mean:.4f}")
                            debug_print(f"Frame {i} Results - Static IoU: {static_mean:.4f}, Dynamic IoU: {dynamic_mean:.4f}", "SUCCESS")
                    else:
                        debug_print(f"Frame {i}: Failed to calculate segmentation metrics", "ERROR")
                    
                    continue
                
                # Object detection task - proceed with normal evaluation
                pred_box_tensor = infer_result_single['pred_box_tensor']
                gt_box_tensor = infer_result_single['gt_box_tensor']
                pred_score = infer_result_single['pred_score']
                
                # Only run evaluation if we have valid tensors (not None or dict)
                if (pred_box_tensor is not None and 
                    gt_box_tensor is not None and 
                    not isinstance(pred_box_tensor, dict) and 
                    not isinstance(gt_box_tensor, dict)):
                    
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat[agent_modality_list[idx]] if opt.all else result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat[agent_modality_list[idx]] if opt.all else result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat[agent_modality_list[idx]] if opt.all else result_stat,
                                            0.7)
                else:
                    print(f"Skipping evaluation - invalid tensor format for modality {agent_modality_list[idx] if opt.all else 'default'}")
                if opt.save_npy:
                    npy_save_path = os.path.join(work_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if not opt.no_score:
                    infer_result_single.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                    infer_result_single.update({"cav_box_np": cav_box_np, \
                                        "agent_modality_list": agent_modality_list})

                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                    vis_save_path_root = os.path.join(work_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                    # simple_vis.visualize(infer_result_single,
                    #                     batch_data['ego'][
                    #                         'origin_lidar'][0],
                    #                     hypes['postprocess']['gt_range'],
                    #                     vis_save_path,
                    #                     method='3d',
                    #                     left_hand=left_hand)
                    
                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(infer_result_single,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand,
                                        show_bev=opt.show_bev)
        torch.cuda.empty_cache()
    
    # Final evaluation results
    if opt.all:
        for modality_name in modality_list:
            work_dir = os.path.join(opt.model_dir, modality_name)
            os.makedirs(work_dir, exist_ok=True)
            
            # Check if we have detection or segmentation results
            if modality_name in seg_result_stat and seg_result_stat[modality_name]['static_iou']:
                # Segmentation evaluation
                static_ious = seg_result_stat[modality_name]['static_iou']
                dynamic_ious = seg_result_stat[modality_name]['dynamic_iou']
                
                avg_static_iou = np.mean([np.mean(iou) for iou in static_ious])
                avg_dynamic_iou = np.mean([np.mean(iou) for iou in dynamic_ious])
                
                print(f"Modality {modality_name} - Segmentation Results:")
                print(f"  Average Static IoU: {avg_static_iou:.4f}")
                print(f"  Average Dynamic IoU: {avg_dynamic_iou:.4f}")
                
                # Save segmentation results
                seg_results = {
                    'static_iou': avg_static_iou,
                    'dynamic_iou': avg_dynamic_iou,
                    'static_ious_per_frame': static_ious,
                    'dynamic_ious_per_frame': dynamic_ious
                }
                
                import json
                with open(os.path.join(work_dir, f'segmentation_results_{infer_info}.json'), 'w') as f:
                    json.dump(seg_results, f, indent=4, default=str)
            else:
                # Detection evaluation
                _, ap50, ap70 = eval_utils.eval_final_results(result_stat[modality_name],
                                            work_dir, infer_info)
    else:
        # Check if we have detection or segmentation results  
        if seg_result_stat.get('static_iou'):
            # Segmentation evaluation
            static_ious = seg_result_stat['static_iou']
            dynamic_ious = seg_result_stat['dynamic_iou']
            
            avg_static_iou = np.mean([np.mean(iou) for iou in static_ious])
            avg_dynamic_iou = np.mean([np.mean(iou) for iou in dynamic_ious])
            
            print(f"\n{'='*80}")
            print(f"🎯 FINAL SEGMENTATION RESULTS")
            print(f"{'='*80}")
            print(f"📊 Legacy IoU Metrics:")
            print(f"  Average Static IoU: {avg_static_iou:.4f}")
            print(f"  Average Dynamic IoU: {avg_dynamic_iou:.4f}")
            
            # Calculate comprehensive metrics if available
            if seg_result_stat.get('static_metrics') and seg_result_stat.get('dynamic_metrics'):
                static_metrics_list = seg_result_stat['static_metrics']
                dynamic_metrics_list = seg_result_stat['dynamic_metrics']
                
                # Average comprehensive metrics across all frames
                avg_static_metrics = {}
                avg_dynamic_metrics = {}
                
                if static_metrics_list:
                    # Calculate averages for static metrics
                    avg_static_metrics = {
                        'mean_iou': np.mean([m['mean_iou'] for m in static_metrics_list if m]),
                        'overall_accuracy': np.mean([m['overall_accuracy'] for m in static_metrics_list if m]),
                        'freq_weighted_iou': np.mean([m['freq_weighted_iou'] for m in static_metrics_list if m]),
                        'mean_class_accuracy': np.mean([m['mean_class_accuracy'] for m in static_metrics_list if m])
                    }
                    
                    # Per-class averages for static
                    class_metrics_avg = {}
                    for class_name in ["Background", "Road", "Lane"]:
                        class_metrics_avg[class_name] = {
                            'iou': np.mean([m['per_class_metrics'][class_name]['iou'] for m in static_metrics_list if m and class_name in m['per_class_metrics']]),
                            'precision': np.mean([m['per_class_metrics'][class_name]['precision'] for m in static_metrics_list if m and class_name in m['per_class_metrics']]),
                            'recall': np.mean([m['per_class_metrics'][class_name]['recall'] for m in static_metrics_list if m and class_name in m['per_class_metrics']]),
                            'f1': np.mean([m['per_class_metrics'][class_name]['f1'] for m in static_metrics_list if m and class_name in m['per_class_metrics']])
                        }
                    avg_static_metrics['per_class_metrics'] = class_metrics_avg
                
                if dynamic_metrics_list:
                    # Calculate averages for dynamic metrics
                    avg_dynamic_metrics = {
                        'mean_iou': np.mean([m['mean_iou'] for m in dynamic_metrics_list if m]),
                        'overall_accuracy': np.mean([m['overall_accuracy'] for m in dynamic_metrics_list if m]),
                        'freq_weighted_iou': np.mean([m['freq_weighted_iou'] for m in dynamic_metrics_list if m]),
                        'mean_class_accuracy': np.mean([m['mean_class_accuracy'] for m in dynamic_metrics_list if m])
                    }
                    
                    # Per-class averages for dynamic
                    class_metrics_avg = {}
                    for class_name in ["Background", "Vehicle"]:
                        class_metrics_avg[class_name] = {
                            'iou': np.mean([m['per_class_metrics'][class_name]['iou'] for m in dynamic_metrics_list if m and class_name in m['per_class_metrics']]),
                            'precision': np.mean([m['per_class_metrics'][class_name]['precision'] for m in dynamic_metrics_list if m and class_name in m['per_class_metrics']]),
                            'recall': np.mean([m['per_class_metrics'][class_name]['recall'] for m in dynamic_metrics_list if m and class_name in m['per_class_metrics']]),
                            'f1': np.mean([m['per_class_metrics'][class_name]['f1'] for m in dynamic_metrics_list if m and class_name in m['per_class_metrics']])
                        }
                    avg_dynamic_metrics['per_class_metrics'] = class_metrics_avg
                
                # Display comprehensive metrics
                print(f"\n🔥 COMPREHENSIVE METRICS:")
                print(f"📈 Static Segmentation:")
                if avg_static_metrics:
                    print(f"   Mean IoU: {avg_static_metrics['mean_iou']:.4f}")
                    print(f"   Overall Accuracy: {avg_static_metrics['overall_accuracy']:.4f}")
                    print(f"   Freq. Weighted IoU: {avg_static_metrics['freq_weighted_iou']:.4f}")
                    print(f"   Mean Class Accuracy: {avg_static_metrics['mean_class_accuracy']:.4f}")
                    print(f"   Per-Class Performance:")
                    for class_name, metrics in avg_static_metrics['per_class_metrics'].items():
                        print(f"     {class_name}: IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
                
                print(f"\n🚗 Dynamic Segmentation:")
                if avg_dynamic_metrics:
                    print(f"   Mean IoU: {avg_dynamic_metrics['mean_iou']:.4f}")
                    print(f"   Overall Accuracy: {avg_dynamic_metrics['overall_accuracy']:.4f}")
                    print(f"   Freq. Weighted IoU: {avg_dynamic_metrics['freq_weighted_iou']:.4f}")
                    print(f"   Mean Class Accuracy: {avg_dynamic_metrics['mean_class_accuracy']:.4f}")
                    print(f"   Per-Class Performance:")
                    for class_name, metrics in avg_dynamic_metrics['per_class_metrics'].items():
                        print(f"     {class_name}: IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
                
                # Vehicle detection summary
                if avg_dynamic_metrics and 'Vehicle' in avg_dynamic_metrics['per_class_metrics']:
                    vehicle_metrics = avg_dynamic_metrics['per_class_metrics']['Vehicle']
                    print(f"\n🎯 VEHICLE DETECTION SUMMARY:")
                    print(f"   Vehicle IoU: {vehicle_metrics['iou']:.4f}")
                    print(f"   Vehicle F1-Score: {vehicle_metrics['f1']:.4f}")
                    print(f"   Vehicle Precision: {vehicle_metrics['precision']:.4f}")
                    print(f"   Vehicle Recall: {vehicle_metrics['recall']:.4f}")
            
            print(f"{'='*80}")
            
            # Save comprehensive segmentation results
            seg_results = {
                'legacy_metrics': {
                    'static_iou': avg_static_iou,
                    'dynamic_iou': avg_dynamic_iou,
                    'static_ious_per_frame': static_ious,
                    'dynamic_ious_per_frame': dynamic_ious
                },
                'comprehensive_metrics': {
                    'static_metrics': avg_static_metrics,
                    'dynamic_metrics': avg_dynamic_metrics,
                    'static_metrics_per_frame': static_metrics_list,
                    'dynamic_metrics_per_frame': dynamic_metrics_list
                }
            }
            
            import json
            with open(os.path.join(opt.model_dir, f'segmentation_results_{infer_info}.json'), 'w') as f:
                json.dump(seg_results, f, indent=4, default=str)
                
            print(f"📁 Detailed results saved to: {opt.model_dir}/segmentation_results_{infer_info}.json")
        else:
            # Detection evaluation
            _, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
