# import os
# import json
#
#
# class Br35Solver(object):
#     CLSNAMES = ['brain']
#
#     def __init__(self, root='data/mvtec'):
#         self.root = root
#         self.meta_path = f'{root}/meta.json'
#
#     def run(self):
#         info = dict(train={}, test={})
#         for cls_name in self.CLSNAMES:
#             cls_dir = f'{self.root}/'
#             for phase in ['test']:
#                 cls_info = []
#                 species = os.listdir(f'{cls_dir}')
#                 for specie in species:
#                     is_abnormal = True if specie not in ['no'] else False
#                     img_names = os.listdir(f'{cls_dir}/{specie}')
#                     img_names.sort()
#                     for idx, img_name in enumerate(img_names):
#                         info_img = dict(
#                             img_path=f'{cls_dir}/{specie}/{img_name}',
#                             cls_name=cls_name,
#                             mask_path="",
#                             specie_name=specie,
#                             anomaly=1 if is_abnormal else 0,
#                         )
#                         cls_info.append(info_img)
#                 info[phase][cls_name] = cls_info
#         with open(self.meta_path, 'w') as f:
#             f.write(json.dumps(info, indent=4) + "\n")
#
# if __name__ == '__main__':
#     runner = Br35Solver(root='/home/wsc2025/AF-CLIP/data/Br35H')
#     runner.run()
import os
import json


class Br35HSolver(object):
    """
    Generate meta.json for Br35H image-level anomaly detection.

    Dataset structure:

    Br35H/
        no/     -> normal brain MRI
        yes/    -> tumor / abnormal brain MRI
        pred/
        Br35H-Mask-RCNN/

    In the current FB-CLIP protocol:
        no  -> anomaly = 0
        yes -> anomaly = 1

    Br35H is treated as an image-level anomaly detection dataset.
    Pixel masks are not used here.
    """

    CLSNAME = "brain"

    VALID_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    def __init__(
        self,
        root="/home/wsc2025/AF-CLIP/data/Br35H"
    ):
        self.root = root

        self.normal_dir = os.path.join(
            self.root,
            "no"
        )

        self.anomaly_dir = os.path.join(
            self.root,
            "yes"
        )

        self.meta_path = os.path.join(
            self.root,
            "meta.json"
        )

    def collect_images(self, folder):
        """
        Recursively collect image files.
        Directories will never be written into meta.json.
        """

        image_paths = []

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Directory does not exist: {folder}"
            )

        for current_root, _, files in os.walk(folder):

            for filename in files:

                if filename.lower().endswith(
                    self.VALID_EXTENSIONS
                ):

                    full_path = os.path.join(
                        current_root,
                        filename
                    )

                    image_paths.append(
                        full_path
                    )

        image_paths.sort()

        return image_paths

    def make_relative_path(self, full_path):
        """
        Convert absolute path to path relative to dataset root.
        """

        relative_path = os.path.relpath(
            full_path,
            self.root
        )

        # Keep json paths platform-independent
        relative_path = relative_path.replace(
            os.sep,
            "/"
        )

        return relative_path

    def run(self):

        print("=" * 70)
        print("Generating Br35H meta.json")
        print("=" * 70)

        # -------------------------------------------------
        # Collect normal images
        # -------------------------------------------------

        normal_images = self.collect_images(
            self.normal_dir
        )

        # -------------------------------------------------
        # Collect anomaly images
        # -------------------------------------------------

        anomaly_images = self.collect_images(
            self.anomaly_dir
        )

        print(
            f"Normal images found : "
            f"{len(normal_images)}"
        )

        print(
            f"Anomaly images found: "
            f"{len(anomaly_images)}"
        )

        # -------------------------------------------------
        # Construct meta information
        # -------------------------------------------------

        info = {
            "train": {
                self.CLSNAME: []
            },
            "test": {
                self.CLSNAME: []
            }
        }

        # -------------------------------------------------
        # Normal samples
        # -------------------------------------------------

        for full_path in normal_images:

            relative_path = self.make_relative_path(
                full_path
            )

            sample = {
                "img_path": relative_path,

                # Br35H image-level dataset:
                # no pixel mask
                "mask_path": "",

                "cls_name": self.CLSNAME,

                "specie_name": "good",

                "anomaly": 0,
            }

            info["test"][
                self.CLSNAME
            ].append(sample)

        # -------------------------------------------------
        # Abnormal samples
        # -------------------------------------------------

        for full_path in anomaly_images:

            relative_path = self.make_relative_path(
                full_path
            )

            sample = {
                "img_path": relative_path,

                # No pixel GT for the image-level Br35H protocol
                "mask_path": "",

                "cls_name": self.CLSNAME,

                "specie_name": "tumor",

                "anomaly": 1,
            }

            info["test"][
                self.CLSNAME
            ].append(sample)

        # -------------------------------------------------
        # Save JSON
        # -------------------------------------------------

        with open(
            self.meta_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                info,
                f,
                indent=4,
                ensure_ascii=False
            )

        print("=" * 70)

        print(
            "Total test samples:",
            len(
                info["test"][
                    self.CLSNAME
                ]
            )
        )

        print(
            "Normal samples:",
            len(normal_images)
        )

        print(
            "Anomaly samples:",
            len(anomaly_images)
        )

        print(
            "Saved to:",
            self.meta_path
        )

        print("=" * 70)


if __name__ == "__main__":

    solver = Br35HSolver(
        root="/home/wsc2025/AF-CLIP/data/Br35H"
    )

    solver.run()