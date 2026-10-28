# import os
# import json
# import pandas as pd
#
#
# class ClinicDBSolver(object):
#     CLSNAMES = [
#         'thyroid',
#     ]
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
#             cls_dir = f'{self.root}'
#             for phase in ['test']:
#                 cls_info = []
#                 # is_abnormal = True if specie not in ['good'] else False
#                 img_names = os.listdir(f'{cls_dir}/test-image')
#                 mask_names = os.listdir(f'{cls_dir}/test-mask')
#                 img_names.sort()
#                 mask_names.sort() if mask_names is not None else None
#                 for idx, img_name in enumerate(img_names):
#                     info_img = dict(
#                         img_path=f'{cls_dir}/test-image/{img_name}',
#                         mask_path=f'{cls_dir}/test-mask/{mask_names[idx]}',
#                         cls_name=cls_name,
#                         specie_name='',
#                         anomaly=1
#                     )
#                     cls_info.append(info_img)
#                     if phase == 'test':
#                         if True:
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
#     runner = ClinicDBSolver(root='/home/wsc2025/AF-CLIP/data/Thyroid_Dataset/tn3k')
#     runner.run()
import os
import json


class TN3KSolver(object):

    CLSNAMES = [
        'tn3k',
    ]

    def __init__(
        self,
        root='/home/wsc2025/AF-CLIP/data/Thyroid_Dataset/tn3k'
    ):
        self.root = root
        self.meta_path = os.path.join(
            root,
            'meta.json'
        )

    def run(self):

        info = {
            'train': {},
            'test': {}
        }

        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:

            img_dir = os.path.join(
                self.root,
                'test-image'
            )

            mask_dir = os.path.join(
                self.root,
                'test-mask'
            )

            img_names = sorted(
                os.listdir(img_dir)
            )

            mask_names = sorted(
                os.listdir(mask_dir)
            )

            print(
                f'Images: {len(img_names)}'
            )

            print(
                f'Masks: {len(mask_names)}'
            )

            if len(img_names) != len(mask_names):

                raise RuntimeError(
                    f'Image/Mask count mismatch: '
                    f'{len(img_names)} images vs '
                    f'{len(mask_names)} masks'
                )

            cls_info = []

            for img_name, mask_name in zip(
                img_names,
                mask_names
            ):

                info_img = {
                    # 使用相对路径
                    'img_path':
                        os.path.join(
                            'test-image',
                            img_name
                        ),

                    'mask_path':
                        os.path.join(
                            'test-mask',
                            mask_name
                        ),

                    'cls_name':
                        cls_name,

                    'specie_name':
                        '',

                    'anomaly':
                        1
                }

                cls_info.append(
                    info_img
                )

                anomaly_samples += 1

            info['test'][cls_name] = (
                cls_info
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

            f.write('\n')

        print(
            f'meta.json saved to: '
            f'{self.meta_path}'
        )

        print(
            'normal_samples:',
            normal_samples
        )

        print(
            'anomaly_samples:',
            anomaly_samples
        )


if __name__ == '__main__':

    runner = TN3KSolver(
        root=
        '/home/wsc2025/AF-CLIP/data/Thyroid_Dataset/tn3k'
    )

    runner.run()