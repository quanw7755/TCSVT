# #!/usr/bin/env python3
# """
# Test with trained model
# Load trained weights and test on specified dataset
# """
#
# import FBCLIP_lib
# import torch
# import argparse
# import torch.nn.functional as F
# from prompt_ensemble import FBCLIP_PromptLearner
# from utils import get_transform, normalize
# from dataset import Dataset
# from logger import get_logger
# from tqdm import tqdm
# import numpy as np
# import os
# import random
# from metrics import image_level_metrics, pixel_level_metrics
# from scipy.ndimage import gaussian_filter
# from tabulate import tabulate
# import matplotlib.pyplot as plt
# import cv2
#
#
#
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# def load_trained_model(model_path, args, device):
#     """Load the trained model"""
#     print(f"Loading the trained model: {model_path}")
#
#     # Check if model file exists
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file does not exist: {model_path}")
#
#     # Model parameters
#     FBCLIP_parameters = {
#         "Prompt_length": args.n_ctx,
#         "learnabel_text_embedding_depth": args.depth,
#         "learnabel_text_embedding_length": args.t_n_ctx
#     }
#
#     # Load model
#     model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=FBCLIP_parameters)
#
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad_(False)
#
#     # Initialize prompt learner
#     prompt_learner = FBCLIP_PromptLearner(model.to("cpu"), FBCLIP_parameters)
#     prompt_learner.to(device)
#     model.to(device)
#     model.FB_params(args=args, device=device)
#
#     # Load trained weights
#     checkpoint = torch.load(model_path, map_location=device)
#
#     # Load prompt learner weights
#     prompt_learner.load_state_dict(checkpoint["prompt_learner"])
#
#     # Load model's trainable parameters
#     if "model_trainable_params" in checkpoint:
#         print("Loading the trainable parameters of the model...")
#         loaded_params = 0
#         for name, param_data in checkpoint["model_trainable_params"].items():
#             # Find corresponding parameter
#             for param_name, param in model.named_parameters():
#                 if param_name == name:
#                     param.data.copy_(param_data)
#                     loaded_params += 1
#                     break
#         print(f"Loaded {loaded_params} trainable parameters from the model")
#     else:
#         print("⚠️  Warning: No trainable parameters were found in the model file. Using default parameters.")
#
#     print(f"Load OK!")
#     print(f"train epoch: {checkpoint.get('epoch', 'Unknown')}")
#     print(f"final loss: {checkpoint.get('loss', 'Unknown')}")
#     print(f"source domain: {checkpoint.get('source_domain', 'Unknown')}")
#
#     return model, prompt_learner
#
# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     """Overlay anomaly score map on original image"""
#     np_image = np.asarray(image, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
#
# def draw_gt_mask_on_image(image, gt_mask):
#     """Draw GT mask region on original image"""
#     # Ensure image is in BGR format
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         img_bgr = image.copy()
#     else:
#         img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Ensure mask is single channel
#     if len(gt_mask.shape) == 3:
#         mask_2d = gt_mask.squeeze()
#     else:
#         mask_2d = gt_mask
#
#     # Convert mask to binary image (0-255)
#     mask_binary = (mask_2d > 127).astype(np.uint8) * 255
#
#     # Find contours
#     contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Draw contours on original image in red
#     annotated_img = img_bgr.copy()
#     cv2.drawContours(annotated_img, contours, -1, (0, 0, 255), thickness=3)
#
#     return annotated_img
#
# def visualize_test_results(image, anomaly_map, gt_mask, save_path, cls_name, sample_idx, dataset_name, anomaly_label):
#     """Visualize test results - reference visualization.py method"""
#     # Determine subfolder based on anomaly status
#     if isinstance(anomaly_label, torch.Tensor):
#         anomaly_value = anomaly_label.item()
#     else:
#         anomaly_value = anomaly_label
#     folder_name = "anomaly" if anomaly_value == 1 else "normal"
#     cls_save_path = os.path.join(save_path, 'imgs', cls_name, folder_name)
#     os.makedirs(cls_save_path, exist_ok=True)
#
#     # Convert image format
#     img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#     img_np = (img_np * 255).astype(np.uint8)
#
#     # Convert anomaly map
#     anomaly_np = anomaly_map.squeeze().cpu().numpy()
#     anomaly_np = normalize(anomaly_np)  # Use normalize function from utils (standard min-max normalization)
#
#     # Convert GT mask
#     gt_np = gt_mask.squeeze().cpu().numpy()
#     gt_np = (gt_np * 255).astype(np.uint8)
#
#     # 1. Save original image
#     original_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_original.png'), original_img)
#
#     # 2. Save Ground Truth
#     cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_gt.png'), gt_np)
#
#     # 3. Save anomaly map (heatmap)
#     anomaly_heatmap = (anomaly_np * 255).astype(np.uint8)
#     anomaly_colored = cv2.applyColorMap(anomaly_heatmap, cv2.COLORMAP_JET)
#     cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_anomaly.png'), anomaly_colored)
#
#     # 4. Save overlay image (original + anomaly map)
#     overlay_img = apply_ad_scoremap(img_np, anomaly_np, alpha=0.5)
#     overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_overlay.png'), overlay_img)
#
#     # 4.5. Save GT annotation on original image (highlight GT region)
#     gt_annotated_img = draw_gt_mask_on_image(img_np, gt_np)
#     cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_gt_annotated.png'), gt_annotated_img)
#
#     # 5. Save 2x2 comparison grid (using matplotlib)
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#
#     # Original image
#     axes[0, 0].imshow(img_np)
#     axes[0, 0].set_title('Original Image')
#     axes[0, 0].axis('off')
#
#     # Ground Truth
#     axes[0, 1].imshow(gt_np, cmap='gray')
#     axes[0, 1].set_title('Ground Truth')
#     axes[0, 1].axis('off')
#
#     # Anomaly map
#     im1 = axes[1, 0].imshow(anomaly_np, cmap='hot')
#     axes[1, 0].set_title('Anomaly Map')
#     axes[1, 0].axis('off')
#     plt.colorbar(im1, ax=axes[1, 0])
#
#     # Overlay
#     axes[1, 1].imshow(overlay_img)
#     axes[1, 1].set_title('Overlay')
#     axes[1, 1].axis('off')
#
#     plt.suptitle(f'{cls_name} - {dataset_name} Test (Sample {sample_idx})')
#     plt.tight_layout()
#
#     # Save comparison grid
#     plt.savefig(os.path.join(cls_save_path, f'{sample_idx:03d}_comparison.png'), dpi=150, bbox_inches='tight')
#     plt.close()
#
# def test_on_dataset(model, prompt_learner, args, dataset_name):
#     """Test model on specified dataset"""
#     print("=" * 60)
#     print(f"Testing on {dataset_name.upper()} dataset")
#     print("=" * 60)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # Data preprocessing
#     preprocess, target_transform = get_transform(args)
#
#     # Test data
#     test_data = Dataset(root=args.test_data_path, transform=preprocess, target_transform=target_transform, dataset_name=dataset_name)
#     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
#
#     # Get class list from test dataset
#     obj_list = test_data.obj_list
#     print(f"{dataset_name.upper()} dataset classes: {obj_list}")
#
#     results = {}
#     for obj in obj_list:
#         results[obj] = {}
#         results[obj]['gt_sp'] = []
#         results[obj]['pr_sp'] = []
#         results[obj]['imgs_masks'] = []
#         results[obj]['anomaly_maps'] = []
#         results[obj]['inference_times'] = []  # Add inference time recording
#
#     print("Starting test...")
#
#     import time  # Import time module
#
#     # Group test data by class - optimized version
#     class_data = {}
#     for batch_items in test_dataloader:
#         batch_size = batch_items['img'].shape[0]
#         cls_names = batch_items['cls_name']
#
#         for i in range(batch_size):
#             cls_name = cls_names[i]
#             if cls_name not in class_data:
#                 class_data[cls_name] = []
#
#             # Create data item for single sample
#             item = {}
#             for key in batch_items:
#                 if isinstance(batch_items[key], torch.Tensor):
#                     item[key] = batch_items[key][i:i+1]  # Keep batch dimension
#                 else:
#                     item[key] = [batch_items[key][i]]
#
#             class_data[cls_name].append(item)
#
#     print(f"Found {len(class_data)} classes, samples per class:")
#     for cls_name, data_list in class_data.items():
#         print(f"  {cls_name}: {len(data_list)} samples")
#
#     # Test each class
#     table_ls = []
#     image_auroc_list = []
#     image_ap_list = []
#     pixel_auroc_list = []
#     pixel_aupro_list = []
#
#     for cls_idx, (cls_name, data_list) in enumerate(class_data.items()):
#         print(f"\n{'='*60}")
#         print(f"Testing class {cls_idx+1}/{len(class_data)}: {cls_name}")
#         print(f"Number of samples: {len(data_list)}")
#         print(f"{'='*60}")
#
#
#
#         # if "cap" not in cls_name:
#         #     continue
#
#         # Test all samples of current class - using batch processing
#         batch_size = args.test_batch_size
#         total_samples = len(data_list)
#
#         print(f"batch_size: {batch_size} ")
#
#         for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Testing {cls_name}"):
#             batch_end = min(batch_start + batch_size, total_samples)
#             batch_items = data_list[batch_start:batch_end]
#             current_batch_size = len(batch_items)
#
#             # Prepare batch data
#             batch_images = []
#             batch_gt_masks = []
#             batch_anomaly_labels = []
#
#             for item in batch_items:
#                 batch_images.append(item['img'])
#                 batch_gt_masks.append(item['img_mask'])
#                 batch_anomaly_labels.append(item['anomaly'])
#
#             # Merge batch data
#             batch_image = torch.cat(batch_images, dim=0).to(device)
#             batch_gt_mask = torch.cat(batch_gt_masks, dim=0)
#             batch_anomaly_label = torch.cat(batch_anomaly_labels, dim=0)
#
#             # Process ground truth mask
#             batch_gt_mask[batch_gt_mask > 0.5] = 1
#             batch_gt_mask[batch_gt_mask <= 0.5] = 0
#
#             # Resize ground truth mask to standard size 518x518 for consistency
#             standard_size = (518, 518)
#             batch_gt_mask_resized = F.interpolate(batch_gt_mask.float(), size=standard_size, mode='nearest')
#             batch_gt_mask_resized = batch_gt_mask_resized.byte()
#
#             # Store ground truth data
#             results[cls_name]['imgs_masks'].append(batch_gt_mask_resized)
#             results[cls_name]['gt_sp'].extend(batch_anomaly_label.detach().cpu())
#
#             with torch.no_grad():
#                 # Start timing
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 start_time = time.time()
#
#                 # Generate parameters using trained prompt learner
#                 prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
#
#                 # Prediction
#                 predict_labels, predict_masks, consistency_loss = model.FB_encode(
#                     batch_image, args=args,
#                     prompts=prompts,
#                     tokenized_prompts=tokenized_prompts,
#                     compound_prompts_text=compound_prompts_text
#                 )
#
#                 # End timing
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 end_time = time.time()
#                 inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
#                 results[cls_name]['inference_times'].append(inference_time)
#
#                 # Get image-level prediction results
#                 text_probs = predict_labels.detach().cpu()
#                 results[cls_name]['pr_sp'].extend(text_probs)
#
#                 # Get pixel-level anomaly maps
#                 anomaly_maps = predict_masks.detach().cpu()
#
#                 # Ensure anomaly map matches ground truth mask size
#                 batch_gt_mask_resized_check = batch_gt_mask.squeeze().cpu()
#                 if len(batch_gt_mask_resized_check.shape) == 3:  # [batch, H, W]
#                     gt_h, gt_w = batch_gt_mask_resized_check.shape[1], batch_gt_mask_resized_check.shape[2]
#                     anomaly_maps = F.interpolate(anomaly_maps, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
#
#                 # Resize all anomaly_maps to standard 518x518 size to ensure consistency for subsequent concatenation
#                 standard_size = (518, 518)
#                 anomaly_maps = F.interpolate(anomaly_maps, size=standard_size, mode='bilinear', align_corners=False)
#
#                 # Apply Gaussian filter (optional)
#                 if args.use_gaussian_filter:
#                     for i in range(anomaly_maps.shape[0]):
#                         anomaly_map_np = anomaly_maps[i].squeeze().cpu().numpy()
#                         anomaly_map_filtered = gaussian_filter(anomaly_map_np, sigma=args.sigma)
#                         anomaly_maps[i] = torch.from_numpy(anomaly_map_filtered).unsqueeze(0)
#
#                 results[cls_name]['anomaly_maps'].append(anomaly_maps)
#
#                 # Visualize results for the first few images
#                 for i in range(min(args.visualize_samples, current_batch_size)):  # Only visualize first few images per batch
#                     item_idx = batch_start + i
#                     if item_idx < args.visualize_samples:  # Only visualize first few images per class
#                         visualize_test_results(
#                             batch_image[i:i+1], anomaly_maps[i:i+1], batch_gt_mask_resized[i:i+1],
#                             args.save_path,
#                             cls_name, item_idx, dataset_name,
#                             batch_anomaly_label[i]  # Pass anomaly label
#                         )
#
#         # Calculate metrics for current class
#         print(f"\nCalculating metrics for {cls_name}...")
#         results[cls_name]['imgs_masks'] = torch.cat(results[cls_name]['imgs_masks'])
#         results[cls_name]['anomaly_maps'] = torch.cat(results[cls_name]['anomaly_maps']).detach().cpu().numpy()
#
#         # Calculate image-level metrics
#         image_auroc = image_level_metrics(results, cls_name, "image-auroc")
#         image_ap = image_level_metrics(results, cls_name, "image-ap")
#         pixel_auroc = pixel_level_metrics(results, cls_name, "pixel-auroc")
#         pixel_aupro = pixel_level_metrics(results, cls_name, "pixel-aupro")
#
#         # Display current class results
#         print(f"\n{cls_name} test results:")
#         print(f"  Image-level AUROC: {image_auroc:.4f} ({image_auroc*100:.1f}%)")
#         print(f"  Image-level AP:    {image_ap:.4f} ({image_ap*100:.1f}%)")
#         print(f"  Pixel-level AUROC: {pixel_auroc:.4f} ({pixel_auroc*100:.1f}%)")
#         print(f"  Pixel-level AUPRO: {pixel_aupro:.4f} ({pixel_aupro*100:.1f}%)")
#
#         # Calculate inference speed for this class
#         cls_times = np.array(results[cls_name]['inference_times'])
#         cls_mean_time = np.mean(cls_times)
#         cls_per_image_time = cls_mean_time / args.test_batch_size
#         cls_throughput = (args.test_batch_size * 1000) / cls_mean_time
#         print(f"  Inference speed: {cls_per_image_time:.2f} ms/image ({cls_throughput:.2f} FPS)")
#
#         # Add to results list
#         table = [cls_name,
#                 f"{pixel_auroc*100:.1f}",
#                 f"{pixel_aupro*100:.1f}",
#                 f"{image_auroc*100:.1f}",
#                 f"{image_ap*100:.1f}"]
#         table_ls.append(table)
#
#         image_auroc_list.append(image_auroc)
#         image_ap_list.append(image_ap)
#         pixel_auroc_list.append(pixel_auroc)
#         pixel_aupro_list.append(pixel_aupro)
#
#     # Calculate final average metrics
#     table_ls.append(['MEAN',
#                     f"{np.mean(pixel_auroc_list)*100:.1f}",
#                     f"{np.mean(pixel_aupro_list)*100:.1f}",
#                     f"{np.mean(image_auroc_list)*100:.1f}",
#                     f"{np.mean(image_ap_list)*100:.1f}"])
#
#     results_table = tabulate(table_ls, headers=['Objects', 'Pixel AUROC', 'Pixel AUPRO', 'Image AUROC', 'Image AP'], tablefmt="pipe")
#
#     # Calculate overall inference speed
#     all_inference_times = []
#     for obj in obj_list:
#         all_inference_times.extend(results[obj]['inference_times'])
#
#     all_times = np.array(all_inference_times)
#     mean_time = np.mean(all_times)
#     std_time = np.std(all_times)
#     per_image_time = mean_time / args.test_batch_size
#     throughput = (args.test_batch_size * 1000) / mean_time
#
#     # Memory statistics
#     current_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
#     peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
#
#     print("\n" + "=" * 80)
#     print(f"{dataset_name.upper()} Dataset Test Results")
#     print("=" * 80)
#     print(results_table)
#
#     print(f"\n{'='*80}")
#     print("Inference Performance Statistics")
#     print(f"{'='*80}")
#     print(f"Average inference time: {mean_time:.2f} ± {std_time:.2f} ms/batch (batch_size={args.test_batch_size})")
#     print(f"Per image time: {per_image_time:.2f} ms")
#     print(f"Throughput: {throughput:.2f} FPS")
#     print(f"Memory usage: {current_memory:.2f} MB (peak: {peak_memory:.2f} MB)")
#     print(f"{'='*80}")
#
#     # Save results
#     result_file = os.path.join(args.save_path, f'{dataset_name}_test_results.txt')
#     with open(result_file, 'w') as f:
#         f.write(f"{dataset_name.upper()} Dataset Test Results\n")
#         f.write("=" * 80 + "\n")
#         f.write(results_table)
#         f.write(f"\n\nDetailed Statistics:\n")
#         f.write(f"Image-level AUROC: {np.mean(image_auroc_list):.4f} ± {np.std(image_auroc_list):.4f}\n")
#         f.write(f"Image-level AP:    {np.mean(image_ap_list):.4f} ± {np.std(image_ap_list):.4f}\n")
#         f.write(f"Pixel-level AUROC: {np.mean(pixel_auroc_list):.4f} ± {np.std(pixel_auroc_list):.4f}\n")
#         f.write(f"Pixel-level AUPRO: {np.mean(pixel_aupro_list):.4f} ± {np.std(pixel_aupro_list):.4f}\n")
#         f.write(f"\nInference Performance Statistics:\n")
#         f.write(f"Average inference time: {mean_time:.2f} ± {std_time:.2f} ms/batch (batch_size={args.test_batch_size})\n")
#         f.write(f"Per image time: {per_image_time:.2f} ms\n")
#         f.write(f"Throughput: {throughput:.2f} FPS\n")
#         f.write(f"Memory usage: {current_memory:.2f} MB (peak: {peak_memory:.2f} MB)\n")
#
#
#     print(f"\nResults saved to: {result_file}")
#     print(f"Visualization results saved to: {os.path.join(args.save_path, 'imgs')}")
#
#     return {
#         'image_auroc': np.mean(image_auroc_list),
#         'image_ap': np.mean(image_ap_list),
#         'pixel_auroc': np.mean(pixel_auroc_list),
#         'pixel_aupro': np.mean(pixel_aupro_list),
#         'mean_per_image_time': per_image_time  # Add average inference time
#     }
#
# def test_with_trained_model(args):
#     """Main function for testing with trained model"""
#
#     # Create save directories
#     os.makedirs(args.save_path, exist_ok=True)
#     os.makedirs(os.path.join(args.save_path, "imgs"), exist_ok=True)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#
#     # Print GPU information
#     if torch.cuda.is_available():
#         print(f"{'='*80}")
#         print(f"GPU info: {torch.cuda.get_device_name(0)}")
#         print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")
#         print(f"{'='*80}")
#
#     print("=" * 80)
#     print("Testing with Trained Model")
#     print("=" * 80)
#     print(f"Model path: {args.model_path}")
#     print(f"Test dataset: {args.test_dataset}")
#     print(f"Test data path: {args.test_data_path}")
#     print(f"Results save path: {args.save_path}")
#
#     # Record initial memory
#     initial_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
#
#     # Load trained model
#     model, prompt_learner = load_trained_model(args.model_path, args, device)
#
#     # Record memory after model loading
#     after_load_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
#     print(f"\nMemory usage after model loading: {after_load_memory:.2f} MB (increase: {after_load_memory - initial_memory:.2f} MB)")
#
#
#     # Test on specified dataset
#     results = test_on_dataset(model, prompt_learner, args, args.test_dataset)
#
#
#     print("\n" + "=" * 80)
#     print("Testing Complete!")
#     print("=" * 80)
#     print(f"Final Average Results:")
#     print(f"  Image-level AUROC: {results['image_auroc']:.4f} ({results['image_auroc']*100:.1f}%)")
#     print(f"  Image-level AP:    {results['image_ap']:.4f} ({results['image_ap']*100:.1f}%)")
#     print(f"  Pixel-level AUROC: {results['pixel_auroc']:.4f} ({results['pixel_auroc']*100:.1f}%)")
#     print(f"  Pixel-level AUPRO: {results['pixel_aupro']:.4f} ({results['pixel_aupro']*100:.1f}%)")
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("Test with Trained Model", add_help=True)
#
#     # 模型和数据路径
#     parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重路径")
#     parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
#     parser.add_argument("--test_dataset", type=str, required=True, help="测试数据集名称")
#     parser.add_argument("--save_path", type=str, default='./trained_model_test_results', help='结果保存路径')
#
#     # 模型参数
#     parser.add_argument("--depth", type=int, default=9, help="learnable text embedding depth")
#     parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
#     parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable text embedding length")
#     parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[5,11,17,24], help="feature map layers")
#     parser.add_argument("--features_list", type=int, nargs="+", default=[5,11,17,24], help="features used")
#     parser.add_argument("--feature_layers", type=int, nargs="+", default=[1,2,3,11,17,24], help="feature layers")
#     parser.add_argument("--image_size", type=int, default=518, help="image size")
#     parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
#
#     # 测试参数
#     parser.add_argument("--test_batch_size", type=int, default=8, help="测试时的批次大小")
#     parser.add_argument("--visualize_samples", type=int, default=3, help="每个类别可视化的样本数量")
#     parser.add_argument("--use_gaussian_filter", action="store_true", help="是否使用高斯滤波")
#     parser.add_argument("--seed", type=int, default=111, help="随机种子")
#
#     args = parser.parse_args()
#
#     print("参数设置:")
#     for arg in vars(args):
#         print(f"  {arg}: {getattr(args, arg)}")
#
#     setup_seed(args.seed)
#     test_with_trained_model(args)
#
# !/usr/bin/env python3
"""
Test with trained model
Load trained weights and test on specified dataset
"""

import FBCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import FBCLIP_PromptLearner
from utils import get_transform, normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
import matplotlib.pyplot as plt
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_entropy(prob, eps=1e-6):
    """Binary entropy for probabilities in [0, 1]."""
    prob = prob.float().clamp(min=eps, max=1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob)).mean()


def restore_prompt_context(prompt_learner, source_ctx_pos, source_ctx_neg):
    """Restore the source-domain shallow prompts for episodic TTA."""
    with torch.no_grad():
        prompt_learner.ctx_pos.copy_(source_ctx_pos)
        prompt_learner.ctx_neg.copy_(source_ctx_neg)


def configure_prompt_tta_parameters(model, prompt_learner):
    """
    Freeze the whole FB-CLIP model and all prompt parameters except
    the shallow normal/abnormal contexts used by the current text path.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for param in prompt_learner.parameters():
        param.requires_grad_(False)

    prompt_learner.ctx_pos.requires_grad_(True)
    prompt_learner.ctx_neg.requires_grad_(True)

    return [prompt_learner.ctx_pos, prompt_learner.ctx_neg]


def fbclip_forward(model, prompt_learner, images, args):
    """Shared FB-CLIP forward used by baseline, adaptation and flip ensemble."""
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)

    predict_labels, predict_masks, consistency_loss = model.FB_encode(
        images,
        args=args,
        prompts=prompts,
        tokenized_prompts=tokenized_prompts,
        compound_prompts_text=compound_prompts_text
    )

    return predict_labels, predict_masks, consistency_loss


def tta_adapt_one_batch(
        model,
        prompt_learner,
        batch_image,
        args,
        source_ctx_pos,
        source_ctx_neg,
):
    """
    Episodic prompt-only test-time adaptation.

    No target labels or masks are used. Each batch starts from the
    source-domain prompt. Only ctx_pos / ctx_neg are updated with:
      1) image-level entropy minimization,
      2) original/flip image-score consistency,
      3) original/flip pixel-map consistency,
      4) source-prompt regularization.
    """
    model.eval()
    prompt_learner.eval()

    tta_params = configure_prompt_tta_parameters(model, prompt_learner)

    optimizer = torch.optim.Adam(
        tta_params,
        lr=args.tta_lr,
        betas=(0.9, 0.999)
    )

    last_stats = {
        "total": 0.0,
        "entropy": 0.0,
        "img_cons": 0.0,
        "pix_cons": 0.0,
        "reg": 0.0,
    }

    for _ in range(args.tta_steps):
        optimizer.zero_grad(set_to_none=True)

        # Original view
        pred_img, pred_map, _ = fbclip_forward(
            model,
            prompt_learner,
            batch_image,
            args
        )

        # Horizontal-flip view
        flipped_image = torch.flip(batch_image, dims=[-1])
        pred_img_flip, pred_map_flip, _ = fbclip_forward(
            model,
            prompt_learner,
            flipped_image,
            args
        )

        # Flip the map back to the original spatial coordinates
        pred_map_flip = torch.flip(pred_map_flip, dims=[-1])

        # Keep entropy weight small for anomaly detection to reduce collapse risk
        loss_entropy = 0.5 * (
                binary_entropy(pred_img) +
                binary_entropy(pred_img_flip)
        )

        # View consistency
        loss_img_cons = F.mse_loss(
            pred_img.float(),
            pred_img_flip.float()
        )

        loss_pix_cons = F.mse_loss(
            pred_map.float(),
            pred_map_flip.float()
        )

        # Keep adapted prompts close to the source-domain prompts
        loss_reg = (
                F.mse_loss(
                    prompt_learner.ctx_pos.float(),
                    source_ctx_pos.float()
                )
                +
                F.mse_loss(
                    prompt_learner.ctx_neg.float(),
                    source_ctx_neg.float()
                )
        )

        loss_tta = (
                args.tta_entropy_weight * loss_entropy
                + args.tta_img_cons_weight * loss_img_cons
                + args.tta_pix_cons_weight * loss_pix_cons
                + args.tta_reg_weight * loss_reg
        )

        if not torch.isfinite(loss_tta):
            raise RuntimeError(
                "TTA loss became NaN/Inf. "
                "Try a smaller --tta_lr or a larger --tta_reg_weight."
            )

        loss_tta.backward()

        torch.nn.utils.clip_grad_norm_(
            tta_params,
            max_norm=args.tta_grad_clip
        )

        optimizer.step()

        last_stats = {
            "total": float(loss_tta.detach().cpu()),
            "entropy": float(loss_entropy.detach().cpu()),
            "img_cons": float(loss_img_cons.detach().cpu()),
            "pix_cons": float(loss_pix_cons.detach().cpu()),
            "reg": float(loss_reg.detach().cpu()),
        }

    prompt_learner.eval()
    return last_stats


@torch.no_grad()
def flip_tta_inference(model, prompt_learner, batch_image, args):
    """
    Test-time augmentation without parameter updates:
    average original-view and horizontal-flip predictions.
    """
    pred_img_1, pred_map_1, _ = fbclip_forward(
        model,
        prompt_learner,
        batch_image,
        args
    )

    flipped_image = torch.flip(batch_image, dims=[-1])

    pred_img_2, pred_map_2, _ = fbclip_forward(
        model,
        prompt_learner,
        flipped_image,
        args
    )

    pred_map_2 = torch.flip(pred_map_2, dims=[-1])

    pred_img = 0.5 * (pred_img_1 + pred_img_2)
    pred_map = 0.5 * (pred_map_1 + pred_map_2)

    return pred_img, pred_map


def load_trained_model(model_path, args, device):
    """Load the trained model"""
    print(f"Loading the trained model: {model_path}")

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # Model parameters
    FBCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # Load model
    model, _ = FBCLIP_lib.load("ViT-L/14@336px", device=device, design_details=FBCLIP_parameters)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Initialize prompt learner
    prompt_learner = FBCLIP_PromptLearner(model.to("cpu"), FBCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.FB_params(args=args, device=device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Load prompt learner weights
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    # Load model's trainable parameters
    if "model_trainable_params" in checkpoint:
        print("Loading the trainable parameters of the model...")
        loaded_params = 0
        for name, param_data in checkpoint["model_trainable_params"].items():
            # Find corresponding parameter
            for param_name, param in model.named_parameters():
                if param_name == name:
                    param.data.copy_(param_data)
                    loaded_params += 1
                    break
        print(f"Loaded {loaded_params} trainable parameters from the model")
    else:
        print("⚠️  Warning: No trainable parameters were found in the model file. Using default parameters.")

    print(f"Load OK!")
    print(f"train epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"final loss: {checkpoint.get('loss', 'Unknown')}")
    print(f"source domain: {checkpoint.get('source_domain', 'Unknown')}")

    # FB_params() creates modules after the first freeze call. Freeze again here.
    # Prompt-only TTA will selectively re-enable ctx_pos / ctx_neg later.
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    prompt_learner.eval()

    return model, prompt_learner


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    """Overlay anomaly score map on original image"""
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def draw_gt_mask_on_image(image, gt_mask):
    """Draw GT mask region on original image"""
    # Ensure image is in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_bgr = image.copy()
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure mask is single channel
    if len(gt_mask.shape) == 3:
        mask_2d = gt_mask.squeeze()
    else:
        mask_2d = gt_mask

    # Convert mask to binary image (0-255)
    mask_binary = (mask_2d > 127).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image in red
    annotated_img = img_bgr.copy()
    cv2.drawContours(annotated_img, contours, -1, (0, 0, 255), thickness=3)

    return annotated_img


def visualize_test_results(image, anomaly_map, gt_mask, save_path, cls_name, sample_idx, dataset_name, anomaly_label):
    """Visualize test results - reference visualization.py method"""
    # Determine subfolder based on anomaly status
    if isinstance(anomaly_label, torch.Tensor):
        anomaly_value = anomaly_label.item()
    else:
        anomaly_value = anomaly_label
    folder_name = "anomaly" if anomaly_value == 1 else "normal"
    cls_save_path = os.path.join(save_path, 'imgs', cls_name, folder_name)
    os.makedirs(cls_save_path, exist_ok=True)

    # Convert image format
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)

    # Convert anomaly map
    anomaly_np = anomaly_map.squeeze().cpu().numpy()
    anomaly_np = normalize(anomaly_np)  # Use normalize function from utils (standard min-max normalization)

    # Convert GT mask
    gt_np = gt_mask.squeeze().cpu().numpy()
    gt_np = (gt_np * 255).astype(np.uint8)

    # 1. Save original image
    original_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_original.png'), original_img)

    # 2. Save Ground Truth
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_gt.png'), gt_np)

    # 3. Save anomaly map (heatmap)
    anomaly_heatmap = (anomaly_np * 255).astype(np.uint8)
    anomaly_colored = cv2.applyColorMap(anomaly_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_anomaly.png'), anomaly_colored)

    # 4. Save overlay image (original + anomaly map)
    overlay_img = apply_ad_scoremap(img_np, anomaly_np, alpha=0.5)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_overlay.png'), overlay_img)

    # 4.5. Save GT annotation on original image (highlight GT region)
    gt_annotated_img = draw_gt_mask_on_image(img_np, gt_np)
    cv2.imwrite(os.path.join(cls_save_path, f'{sample_idx:03d}_gt_annotated.png'), gt_annotated_img)

    # 5. Save 2x2 comparison grid (using matplotlib)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Ground Truth
    axes[0, 1].imshow(gt_np, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    # Anomaly map
    im1 = axes[1, 0].imshow(anomaly_np, cmap='hot')
    axes[1, 0].set_title('Anomaly Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])

    # Overlay
    axes[1, 1].imshow(overlay_img)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    plt.suptitle(f'{cls_name} - {dataset_name} Test (Sample {sample_idx})')
    plt.tight_layout()

    # Save comparison grid
    plt.savefig(os.path.join(cls_save_path, f'{sample_idx:03d}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def test_on_dataset(model, prompt_learner, args, dataset_name):
    """Test model on the specified dataset, optionally with TTA."""
    print("=" * 60)
    print(f"Testing on {dataset_name.upper()} dataset")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    prompt_learner.eval()

    # Data preprocessing
    preprocess, target_transform = get_transform(args)

    # Test data
    test_data = Dataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False
    )

    # Get class list from test dataset
    obj_list = test_data.obj_list
    print(f"{dataset_name.upper()} dataset classes: {obj_list}")

    results = {}
    for obj in obj_list:
        results[obj] = {
            'gt_sp': [],
            'pr_sp': [],
            'imgs_masks': [],
            'anomaly_maps': [],
            'inference_times': [],
            'inference_batch_sizes': [],
            'tta_losses': [],
        }

    print("Starting test...")

    import time

    # Source prompt snapshots for episodic TTA.
    # Every test batch is adapted independently from this same source state.
    source_ctx_pos = prompt_learner.ctx_pos.detach().clone()
    source_ctx_neg = prompt_learner.ctx_neg.detach().clone()

    if args.use_tta:
        print("TTA mode: episodic prompt-only adaptation")
        print(
            f"  steps={args.tta_steps}, lr={args.tta_lr}, "
            f"entropy_w={args.tta_entropy_weight}, "
            f"img_cons_w={args.tta_img_cons_weight}, "
            f"pix_cons_w={args.tta_pix_cons_weight}, "
            f"reg_w={args.tta_reg_weight}"
        )
    if args.use_tta_aug:
        print("Test-time augmentation: original + horizontal flip ensemble")

    # Group test data by class
    class_data = {}
    for batch_items in test_dataloader:
        batch_size = batch_items['img'].shape[0]
        cls_names = batch_items['cls_name']

        for i in range(batch_size):
            cls_name = cls_names[i]
            if cls_name not in class_data:
                class_data[cls_name] = []

            item = {}
            for key in batch_items:
                if isinstance(batch_items[key], torch.Tensor):
                    item[key] = batch_items[key][i:i + 1]
                else:
                    item[key] = [batch_items[key][i]]

            class_data[cls_name].append(item)

    print(f"Found {len(class_data)} classes, samples per class:")
    for cls_name, data_list in class_data.items():
        print(f"  {cls_name}: {len(data_list)} samples")

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    standard_size = (args.image_size, args.image_size)

    for cls_idx, (cls_name, data_list) in enumerate(class_data.items()):
        print(f"\n{'=' * 60}")
        print(f"Testing class {cls_idx + 1}/{len(class_data)}: {cls_name}")
        print(f"Number of samples: {len(data_list)}")
        print(f"{'=' * 60}")

        batch_size = args.test_batch_size
        total_samples = len(data_list)
        print(f"batch_size: {batch_size}")

        for batch_start in tqdm(
                range(0, total_samples, batch_size),
                desc=f"Testing {cls_name}"
        ):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_items = data_list[batch_start:batch_end]
            current_batch_size = len(batch_items)

            batch_images = []
            batch_gt_masks = []
            batch_anomaly_labels = []

            for item in batch_items:
                batch_images.append(item['img'])
                batch_gt_masks.append(item['img_mask'])
                batch_anomaly_labels.append(item['anomaly'])

            batch_image = torch.cat(batch_images, dim=0).to(device)
            batch_gt_mask = torch.cat(batch_gt_masks, dim=0)
            batch_anomaly_label = torch.cat(batch_anomaly_labels, dim=0)

            batch_gt_mask[batch_gt_mask > 0.5] = 1
            batch_gt_mask[batch_gt_mask <= 0.5] = 0

            batch_gt_mask_resized = F.interpolate(
                batch_gt_mask.float(),
                size=standard_size,
                mode='nearest'
            ).byte()

            results[cls_name]['imgs_masks'].append(batch_gt_mask_resized)
            results[cls_name]['gt_sp'].extend(batch_anomaly_label.detach().cpu())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # ----------------------------------------------------------
            # 1) Episodic prompt-only Test-Time Adaptation
            # ----------------------------------------------------------
            if args.use_tta:
                # Always restart from the source-domain prompt.
                restore_prompt_context(
                    prompt_learner,
                    source_ctx_pos,
                    source_ctx_neg
                )

                with torch.enable_grad():
                    tta_stats = tta_adapt_one_batch(
                        model,
                        prompt_learner,
                        batch_image,
                        args,
                        source_ctx_pos,
                        source_ctx_neg,
                    )
                results[cls_name]['tta_losses'].append(tta_stats['total'])

            # ----------------------------------------------------------
            # 2) Final prediction
            # ----------------------------------------------------------
            model.eval()
            prompt_learner.eval()

            with torch.no_grad():
                if args.use_tta_aug:
                    predict_labels, predict_masks = flip_tta_inference(
                        model,
                        prompt_learner,
                        batch_image,
                        args
                    )
                else:
                    predict_labels, predict_masks, _ = fbclip_forward(
                        model,
                        prompt_learner,
                        batch_image,
                        args
                    )

            # Episodic protocol: do not carry target-batch updates forward.
            if args.use_tta:
                restore_prompt_context(
                    prompt_learner,
                    source_ctx_pos,
                    source_ctx_neg
                )
                prompt_learner.ctx_pos.requires_grad_(False)
                prompt_learner.ctx_neg.requires_grad_(False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            # TTA time is intentionally included so the reported speed is honest.
            inference_time = (end_time - start_time) * 1000.0
            results[cls_name]['inference_times'].append(inference_time)
            results[cls_name]['inference_batch_sizes'].append(current_batch_size)

            # Image-level predictions
            text_probs = predict_labels.detach().cpu()
            results[cls_name]['pr_sp'].extend(text_probs)

            # Pixel-level anomaly maps
            anomaly_maps = predict_masks.detach().cpu()

            # First align to the GT size if needed
            batch_gt_mask_check = batch_gt_mask.squeeze().cpu()
            if len(batch_gt_mask_check.shape) == 3:
                gt_h, gt_w = batch_gt_mask_check.shape[1], batch_gt_mask_check.shape[2]
                anomaly_maps = F.interpolate(
                    anomaly_maps,
                    size=(gt_h, gt_w),
                    mode='bilinear',
                    align_corners=False
                )

            # Then use one common size for concatenation/evaluation
            anomaly_maps = F.interpolate(
                anomaly_maps,
                size=standard_size,
                mode='bilinear',
                align_corners=False
            )

            if args.use_gaussian_filter:
                for i in range(anomaly_maps.shape[0]):
                    anomaly_map_np = anomaly_maps[i].squeeze().numpy()
                    anomaly_map_filtered = gaussian_filter(
                        anomaly_map_np,
                        sigma=args.sigma
                    )
                    anomaly_maps[i] = torch.from_numpy(
                        anomaly_map_filtered
                    ).unsqueeze(0)

            results[cls_name]['anomaly_maps'].append(anomaly_maps)

            # Visualize only the first N samples of each class
            for i in range(min(args.visualize_samples, current_batch_size)):
                item_idx = batch_start + i
                if item_idx < args.visualize_samples:
                    visualize_test_results(
                        batch_image[i:i + 1],
                        anomaly_maps[i:i + 1],
                        batch_gt_mask_resized[i:i + 1],
                        args.save_path,
                        cls_name,
                        item_idx,
                        dataset_name,
                        batch_anomaly_label[i]
                    )

        # --------------------------------------------------------------
        # Metrics for the current class
        # --------------------------------------------------------------
        print(f"\nCalculating metrics for {cls_name}...")
        results[cls_name]['imgs_masks'] = torch.cat(results[cls_name]['imgs_masks'])
        results[cls_name]['anomaly_maps'] = torch.cat(
            results[cls_name]['anomaly_maps']
        ).detach().cpu().numpy()

        image_auroc = image_level_metrics(results, cls_name, "image-auroc")
        image_ap = image_level_metrics(results, cls_name, "image-ap")
        pixel_auroc = pixel_level_metrics(results, cls_name, "pixel-auroc")
        pixel_aupro = pixel_level_metrics(results, cls_name, "pixel-aupro")

        print(f"\n{cls_name} test results:")
        print(f"  Image-level AUROC: {image_auroc:.4f} ({image_auroc * 100:.1f}%)")
        print(f"  Image-level AP:    {image_ap:.4f} ({image_ap * 100:.1f}%)")
        print(f"  Pixel-level AUROC: {pixel_auroc:.4f} ({pixel_auroc * 100:.1f}%)")
        print(f"  Pixel-level AUPRO: {pixel_aupro:.4f} ({pixel_aupro * 100:.1f}%)")

        cls_times = np.asarray(results[cls_name]['inference_times'], dtype=np.float64)
        cls_batch_sizes = np.asarray(
            results[cls_name]['inference_batch_sizes'],
            dtype=np.float64
        )
        cls_total_time = cls_times.sum()
        cls_total_images = cls_batch_sizes.sum()
        cls_per_image_time = cls_total_time / max(cls_total_images, 1.0)
        cls_throughput = (
            cls_total_images * 1000.0 / cls_total_time
            if cls_total_time > 0 else 0.0
        )
        print(
            f"  Inference speed: {cls_per_image_time:.2f} ms/image "
            f"({cls_throughput:.2f} FPS)"
        )

        if args.use_tta and results[cls_name]['tta_losses']:
            print(
                f"  Mean TTA loss: "
                f"{np.mean(results[cls_name]['tta_losses']):.6f}"
            )

        table_ls.append([
            cls_name,
            f"{pixel_auroc * 100:.1f}",
            f"{pixel_aupro * 100:.1f}",
            f"{image_auroc * 100:.1f}",
            f"{image_ap * 100:.1f}"
        ])

        image_auroc_list.append(image_auroc)
        image_ap_list.append(image_ap)
        pixel_auroc_list.append(pixel_auroc)
        pixel_aupro_list.append(pixel_aupro)

    # Restore source prompt once more before leaving evaluation
    restore_prompt_context(prompt_learner, source_ctx_pos, source_ctx_neg)
    prompt_learner.eval()

    # Final average metrics
    table_ls.append([
        'MEAN',
        f"{np.mean(pixel_auroc_list) * 100:.1f}",
        f"{np.mean(pixel_aupro_list) * 100:.1f}",
        f"{np.mean(image_auroc_list) * 100:.1f}",
        f"{np.mean(image_ap_list) * 100:.1f}"
    ])

    results_table = tabulate(
        table_ls,
        headers=['Objects', 'Pixel AUROC', 'Pixel AUPRO', 'Image AUROC', 'Image AP'],
        tablefmt="pipe"
    )

    all_inference_times = []
    all_batch_sizes = []
    for obj in obj_list:
        all_inference_times.extend(results[obj]['inference_times'])
        all_batch_sizes.extend(results[obj]['inference_batch_sizes'])

    all_times = np.asarray(all_inference_times, dtype=np.float64)
    all_sizes = np.asarray(all_batch_sizes, dtype=np.float64)

    mean_time = float(np.mean(all_times)) if all_times.size else 0.0
    std_time = float(np.std(all_times)) if all_times.size else 0.0
    total_time = float(np.sum(all_times)) if all_times.size else 0.0
    total_images = float(np.sum(all_sizes)) if all_sizes.size else 0.0
    per_image_time = total_time / max(total_images, 1.0)
    throughput = total_images * 1000.0 / total_time if total_time > 0 else 0.0

    current_memory = (
        torch.cuda.memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0
    )
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0
    )

    print("\n" + "=" * 80)
    print(f"{dataset_name.upper()} Dataset Test Results")
    print("=" * 80)
    print(results_table)

    print(f"\n{'=' * 80}")
    print("Inference Performance Statistics")
    print(f"{'=' * 80}")
    print(
        f"Average inference time: {mean_time:.2f} ± {std_time:.2f} ms/batch "
        f"(nominal batch_size={args.test_batch_size})"
    )
    print(f"Per image time: {per_image_time:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    print(f"Memory usage: {current_memory:.2f} MB (peak: {peak_memory:.2f} MB)")
    print(f"{'=' * 80}")

    result_file = os.path.join(
        args.save_path,
        f'{dataset_name}_test_results.txt'
    )

    with open(result_file, 'w') as f:
        f.write(f"{dataset_name.upper()} Dataset Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(results_table)
        f.write("\n\nDetailed Statistics:\n")
        f.write(
            f"Image-level AUROC: {np.mean(image_auroc_list):.4f} ± "
            f"{np.std(image_auroc_list):.4f}\n"
        )
        f.write(
            f"Image-level AP:    {np.mean(image_ap_list):.4f} ± "
            f"{np.std(image_ap_list):.4f}\n"
        )
        f.write(
            f"Pixel-level AUROC: {np.mean(pixel_auroc_list):.4f} ± "
            f"{np.std(pixel_auroc_list):.4f}\n"
        )
        f.write(
            f"Pixel-level AUPRO: {np.mean(pixel_aupro_list):.4f} ± "
            f"{np.std(pixel_aupro_list):.4f}\n"
        )
        f.write("\nInference Performance Statistics:\n")
        f.write(
            f"Average inference time: {mean_time:.2f} ± {std_time:.2f} ms/batch "
            f"(nominal batch_size={args.test_batch_size})\n"
        )
        f.write(f"Per image time: {per_image_time:.2f} ms\n")
        f.write(f"Throughput: {throughput:.2f} FPS\n")
        f.write(
            f"Memory usage: {current_memory:.2f} MB "
            f"(peak: {peak_memory:.2f} MB)\n"
        )

        f.write("\nTest-Time Configuration:\n")
        f.write(f"use_tta: {args.use_tta}\n")
        f.write(f"use_tta_aug: {args.use_tta_aug}\n")
        if args.use_tta:
            f.write(f"tta_steps: {args.tta_steps}\n")
            f.write(f"tta_lr: {args.tta_lr}\n")
            f.write(f"tta_entropy_weight: {args.tta_entropy_weight}\n")
            f.write(f"tta_img_cons_weight: {args.tta_img_cons_weight}\n")
            f.write(f"tta_pix_cons_weight: {args.tta_pix_cons_weight}\n")
            f.write(f"tta_reg_weight: {args.tta_reg_weight}\n")
            f.write(f"tta_grad_clip: {args.tta_grad_clip}\n")

    print(f"\nResults saved to: {result_file}")
    print(f"Visualization results saved to: {os.path.join(args.save_path, 'imgs')}")

    return {
        'image_auroc': np.mean(image_auroc_list),
        'image_ap': np.mean(image_ap_list),
        'pixel_auroc': np.mean(pixel_auroc_list),
        'pixel_aupro': np.mean(pixel_aupro_list),
        'mean_per_image_time': per_image_time
    }


def test_with_trained_model(args):
    """Main function for testing with trained model"""

    # Create save directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "imgs"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Print GPU information
    if torch.cuda.is_available():
        print(f"{'=' * 80}")
        print(f"GPU info: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")
        print(f"{'=' * 80}")

    print("=" * 80)
    print("Testing with Trained Model")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Test data path: {args.test_data_path}")
    print(f"Results save path: {args.save_path}")

    # Record initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # Load trained model
    model, prompt_learner = load_trained_model(args.model_path, args, device)

    # Record memory after model loading
    after_load_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(
        f"\nMemory usage after model loading: {after_load_memory:.2f} MB (increase: {after_load_memory - initial_memory:.2f} MB)")

    # Test on specified dataset
    results = test_on_dataset(model, prompt_learner, args, args.test_dataset)

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print(f"Final Average Results:")
    print(f"  Image-level AUROC: {results['image_auroc']:.4f} ({results['image_auroc'] * 100:.1f}%)")
    print(f"  Image-level AP:    {results['image_ap']:.4f} ({results['image_ap'] * 100:.1f}%)")
    print(f"  Pixel-level AUROC: {results['pixel_auroc']:.4f} ({results['pixel_auroc'] * 100:.1f}%)")
    print(f"  Pixel-level AUPRO: {results['pixel_aupro']:.4f} ({results['pixel_aupro'] * 100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test with Trained Model", add_help=True)

    # 模型和数据路径
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--test_dataset", type=str, required=True, help="测试数据集名称")
    parser.add_argument("--save_path", type=str, default='./trained_model_test_results', help='结果保存路径')

    # 模型参数
    parser.add_argument("--depth", type=int, default=9, help="learnable text embedding depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable text embedding length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[5, 11, 17, 24], help="feature map layers")
    parser.add_argument("--features_list", type=int, nargs="+", default=[5, 11, 17, 24], help="features used")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[1, 2, 3, 11, 17, 24], help="feature layers")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")

    # 测试参数
    parser.add_argument("--test_batch_size", type=int, default=8, help="测试时的批次大小")
    parser.add_argument("--visualize_samples", type=int, default=3, help="每个类别可视化的样本数量")
    parser.add_argument("--use_gaussian_filter", action="store_true", help="是否使用高斯滤波")

    # Test-Time Adaptation / Test-Time Augmentation
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="启用逐batch重置的prompt-only测试时自适应"
    )
    parser.add_argument(
        "--use_tta_aug",
        action="store_true",
        help="最终预测使用原图+水平翻转的测试时增强集成"
    )
    parser.add_argument("--tta_steps", type=int, default=1, help="每个batch的TTA更新步数")
    parser.add_argument("--tta_lr", type=float, default=5e-5, help="TTA学习率")
    parser.add_argument("--tta_entropy_weight", type=float, default=0.05, help="图像级熵最小化权重")
    parser.add_argument("--tta_img_cons_weight", type=float, default=1.0, help="图像级翻转一致性权重")
    parser.add_argument("--tta_pix_cons_weight", type=float, default=1.0, help="像素级翻转一致性权重")
    parser.add_argument("--tta_reg_weight", type=float, default=0.1, help="源域prompt保持正则权重")
    parser.add_argument("--tta_grad_clip", type=float, default=1.0, help="TTA梯度裁剪阈值")

    parser.add_argument("--seed", type=int, default=111, help="随机种子")

    args = parser.parse_args()

    print("参数设置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    setup_seed(args.seed)
    test_with_trained_model(args)
