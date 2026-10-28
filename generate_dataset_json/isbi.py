# import os
# import json
#
#
# class IsbiSolver(object):
#     CLSNAMES = ['skin']
#
#     def __init__(self, root='data/mvtec'):
#         self.root = root
#         self.meta_path = f'{root}/meta.json'
#
#     def run(self):
#         info = dict(train={}, test={})
#         anomaly_samples = 0
#         normal_samples = 0
#         for cls_name in self.CLSNAMES:
#             cls_dir = f'{self.root}/{cls_name}'
#             for phase in ['test']:
#                 cls_info = []
#                 species = os.listdir(f'{cls_dir}/{phase}')
#                 for specie in species:
#                     is_abnormal = True if specie not in ['good'] else False
#                     img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
#                     mask_names = os.listdir(f'{cls_dir}/ISBI2016_ISIC_Part1_Test_GroundTruth/') if is_abnormal else None
#                     img_names.sort()
#                     mask_names.sort() if mask_names is not None else None
#                     for idx, img_name in enumerate(img_names):
#                         info_img = dict(
#                             img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
#                             mask_path=f'{cls_name}/ISBI2016_ISIC_Part1_Test_GroundTruth/{mask_names[idx]}' if is_abnormal else '',
#                             cls_name=cls_name,
#                             specie_name=specie,
#                             anomaly=1 if is_abnormal else 0,
#                         )
#                         cls_info.append(info_img)
#                         if phase == 'test':
#                             if is_abnormal:
#                                 anomaly_samples = anomaly_samples + 1
#                             else:
#                                 normal_samples = normal_samples + 1
#                 info[phase][cls_name] = cls_info
#         with open(self.meta_path, 'w') as f:
#             f.write(json.dumps(info, indent=4) + "\n")
#         print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
#
# if __name__ == '__main__':
#     runner = IsbiSolver(root='/home/wsc2025/AF-CLIP/data/ISIC2016')
#     runner.run()
#
#
import os
import json
import numpy as np
from PIL import Image


class IsbiSolver(object):

    CLSNAMES = ['skin']

    def __init__(self, root):
        self.root = root

        self.image_dir = os.path.join(
            root,
            'ISBI2016_ISIC_Part1_Test_Data'
        )

        self.mask_dir = os.path.join(
            root,
            'ISBI2016_ISIC_Part1_Test_GroundTruth'
        )

        self.meta_path = os.path.join(
            root,
            'meta.json'
        )

    def find_mask(self, img_name):

        stem = os.path.splitext(img_name)[0]

        # ISIC2016 常见的GT命名方式
        candidates = [
            stem + '_Segmentation.png',
            stem + '_segmentation.png',
            stem + '_mask.png',
            stem + '.png',
        ]

        for mask_name in candidates:
            mask_path = os.path.join(
                self.mask_dir,
                mask_name
            )

            if os.path.isfile(mask_path):
                return mask_name

        return None

    def run(self):

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f'Image directory not found: '
                f'{self.image_dir}'
            )

        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f'Ground-truth directory not found: '
                f'{self.mask_dir}'
            )

        info = {
            'train': {
                'skin': []
            },
            'test': {
                'skin': []
            }
        }

        normal_samples = 0
        anomaly_samples = 0
        missing_masks = 0

        img_names = sorted([
            name
            for name in os.listdir(self.image_dir)
            if name.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.bmp')
            )
        ])

        print(
            f'Found {len(img_names)} test images.'
        )

        for img_name in img_names:

            mask_name = self.find_mask(
                img_name
            )

            if mask_name is None:
                print(
                    f'[WARNING] Mask not found: '
                    f'{img_name}'
                )

                missing_masks += 1
                continue

            mask_abs_path = os.path.join(
                self.mask_dir,
                mask_name
            )

            # 根据 mask 是否存在前景确定 anomaly
            mask = Image.open(
                mask_abs_path
            ).convert('L')

            mask_np = np.array(mask)

            is_abnormal = np.any(
                mask_np > 0
            )

            if is_abnormal:
                anomaly = 1
                anomaly_samples += 1
                specie_name = 'lesion'

                mask_path = (
                    'ISBI2016_ISIC_Part1_'
                    'Test_GroundTruth/'
                    f'{mask_name}'
                )

            else:
                anomaly = 0
                normal_samples += 1
                specie_name = 'good'

                # normal image 在 Dataset 中
                # 会自动生成全0 mask
                mask_path = ''

            info_img = {
                'img_path': (
                    'ISBI2016_ISIC_Part1_'
                    'Test_Data/'
                    f'{img_name}'
                ),

                'mask_path': mask_path,

                'cls_name': 'skin',

                'specie_name': specie_name,

                'anomaly': anomaly,
            }

            info['test']['skin'].append(
                info_img
            )

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

        print()
        print('=' * 60)
        print('ISIC2016 meta.json generated')
        print('=' * 60)

        print(
            'Total samples   :',
            len(info['test']['skin'])
        )

        print(
            'Normal samples  :',
            normal_samples
        )

        print(
            'Anomaly samples :',
            anomaly_samples
        )

        print(
            'Missing masks   :',
            missing_masks
        )

        print(
            'Saved to        :',
            self.meta_path
        )

        print('=' * 60)


if __name__ == '__main__':

    runner = IsbiSolver(
        root='/home/wsc2025/AF-CLIP/data/ISIC2016'
    )

    runner.run()