# import os
# import json
# import pandas as pd
#
#
# class DAGMSolver(object):
#     CLSNAMES = [
#         'Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10'
#     ]
#
#     def __init__(self, root='data/mvtec'):
#         self.root = root
#         self.meta_path = f'{root}/meta.json'
#
#     def run(self):
#         info = dict(Train={}, Test={})
#         anomaly_samples = 0
#         normal_samples = 0
#         for cls_name in self.CLSNAMES:
#             cls_dir = f'{self.root}/{cls_name}'
#             for phase in ['Train', 'Test']:
#                 cls_info = []
#                 x, y, mask_names_none= [], [], []
#                 img_dir = os.listdir(f'{cls_dir}/{phase}')
#
#                 mask_names = os.listdir(f'{cls_dir}/{phase}/Label')
#
#                 img_fpath_list = sorted([f
#                                         for f in img_dir
#                                         if f.endswith('.PNG')])
#                 gt_fpath_list = sorted([f
#                             for f in mask_names
#                             if f.endswith('.PNG')])
#
#                 img_exclude_list = [f.split("_")[0] + ".PNG" for f in gt_fpath_list]
#
#                 img_normal_fpath_list = list(set(img_fpath_list) - set(img_exclude_list))
#
#                 x.extend(img_normal_fpath_list + img_exclude_list)
#
#                 y.extend([0] * len(img_normal_fpath_list) + [1]* len(img_exclude_list))
#
#                 mask_names_none.extend([None] * len(img_normal_fpath_list) + gt_fpath_list)
#
#                 for idx, img_name in enumerate(x):
#                     info_img = dict(
#                         img_path=f'{cls_name}/{phase}/{img_name}',
#                         mask_path=f'{cls_name}/{phase}/Label/{mask_names_none[idx]}' if mask_names_none[idx] != None else '',
#                         cls_name=cls_name,
#                         specie_name='',
#                         anomaly=1 if y[idx] == 1 else 0,
#                     )
#                     cls_info.append(info_img)
#                     if phase == 'Test':
#                         if y[idx] == 1:
#                             anomaly_samples = anomaly_samples + 1
#                         else:
#                             normal_samples = normal_samples + 1
#                 info[phase][cls_name] = cls_info
#         with open(self.meta_path, 'w') as f:
#             f.write(json.dumps(info, indent=4) + "\n")
#         print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
#
#
#
# if __name__ == '__main__':
#     runner = DAGMSolver(root='/home/wsc2025/AF-CLIP/data/DAGM_KaggleUpload')
#     runner.run()
import os
import json


class DAGMSolver(object):

    CLSNAMES = [
        'Class1',
        'Class2',
        'Class3',
        'Class4',
        'Class5',
        'Class6',
        'Class7',
        'Class8',
        'Class9',
        'Class10',
    ]

    def __init__(
        self,
        root='/home/wsc2025/AF-CLIP/data/DAGM_KaggleUpload'
    ):
        self.root = root
        self.meta_path = os.path.join(
            root,
            'meta.json'
        )

    def run(self):

        # ====================================================
        # meta.json 使用小写 train/test
        # 与 Dataset(mode='train'/'test') 保持一致
        # ====================================================
        info = {
            'train': {},
            'test': {}
        }

        # 实际 DAGM 文件夹仍为 Train/Test
        folder_map = {
            'train': 'Train',
            'test': 'Test'
        }

        anomaly_samples = {
            'train': 0,
            'test': 0
        }

        normal_samples = {
            'train': 0,
            'test': 0
        }

        for cls_name in self.CLSNAMES:

            cls_dir = os.path.join(
                self.root,
                cls_name
            )

            print(
                f'\nProcessing {cls_name}'
            )

            for split in ['train', 'test']:

                phase = folder_map[split]

                phase_dir = os.path.join(
                    cls_dir,
                    phase
                )

                label_dir = os.path.join(
                    phase_dir,
                    'Label'
                )

                # ============================================
                # 基础目录检查
                # ============================================
                if not os.path.isdir(phase_dir):
                    raise FileNotFoundError(
                        f'Phase directory not found: '
                        f'{phase_dir}'
                    )

                if not os.path.isdir(label_dir):
                    raise FileNotFoundError(
                        f'Label directory not found: '
                        f'{label_dir}'
                    )

                # ============================================
                # 获取图像
                # ============================================
                img_names = sorted([
                    f
                    for f in os.listdir(phase_dir)
                    if os.path.isfile(
                        os.path.join(
                            phase_dir,
                            f
                        )
                    )
                    and f.lower().endswith('.png')
                ])

                # ============================================
                # 获取 masks
                # ============================================
                mask_names = sorted([
                    f
                    for f in os.listdir(label_dir)
                    if os.path.isfile(
                        os.path.join(
                            label_dir,
                            f
                        )
                    )
                    and f.lower().endswith('.png')
                ])

                # ============================================
                # 建立 image -> mask 映射
                #
                # DAGM 通常:
                # image: 0001.PNG
                # mask : 0001_label.PNG
                # ============================================
                mask_map = {}

                for mask_name in mask_names:

                    image_stem = (
                        mask_name.split('_')[0]
                    )

                    image_name = (
                        image_stem + '.PNG'
                    )

                    mask_map[
                        image_name.lower()
                    ] = mask_name

                cls_info = []

                class_normal = 0
                class_anomaly = 0

                # ============================================
                # 按实际 image 遍历
                # ============================================
                for img_name in img_names:

                    img_key = img_name.lower()

                    if img_key in mask_map:

                        # ----------------------------
                        # abnormal sample
                        # ----------------------------
                        anomaly = 1

                        mask_name = (
                            mask_map[img_key]
                        )

                        mask_path = os.path.join(
                            cls_name,
                            phase,
                            'Label',
                            mask_name
                        )

                        class_anomaly += 1

                    else:

                        # ----------------------------
                        # normal sample
                        # ----------------------------
                        anomaly = 0
                        mask_path = ''

                        class_normal += 1

                    # ========================================
                    # 使用相对路径
                    # ========================================
                    img_path = os.path.join(
                        cls_name,
                        phase,
                        img_name
                    )

                    info_img = {
                        'img_path':
                            img_path,

                        'mask_path':
                            mask_path,

                        'cls_name':
                            cls_name,

                        'specie_name':
                            '',

                        'anomaly':
                            anomaly,
                    }

                    cls_info.append(
                        info_img
                    )

                info[split][
                    cls_name
                ] = cls_info

                normal_samples[
                    split
                ] += class_normal

                anomaly_samples[
                    split
                ] += class_anomaly

                print(
                    f'  {phase}: '
                    f'total={len(cls_info)}, '
                    f'normal={class_normal}, '
                    f'anomaly={class_anomaly}'
                )

        # ====================================================
        # 保存 meta.json
        # ====================================================
        with open(
            self.meta_path,
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                info,
                f,
                indent=4,
                ensure_ascii=False
            )

            f.write('\n')

        print('\n' + '=' * 60)
        print('DAGM meta.json generated successfully')
        print('=' * 60)

        print(
            f"Train: "
            f"normal={normal_samples['train']}, "
            f"anomaly={anomaly_samples['train']}"
        )

        print(
            f"Test : "
            f"normal={normal_samples['test']}, "
            f"anomaly={anomaly_samples['test']}"
        )

        print(
            f'meta.json: {self.meta_path}'
        )


if __name__ == '__main__':

    runner = DAGMSolver(
        root=
        '/home/wsc2025/AF-CLIP/data/DAGM_KaggleUpload'
    )

    runner.run()