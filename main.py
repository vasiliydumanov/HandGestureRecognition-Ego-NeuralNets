import coremltools
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from keras.applications import MobileNetV2
from keras.callbacks import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
from tqdm import tqdm
from collections import namedtuple

from DataGenerator import DataGenerator
import coremltools.proto.FeatureTypes_pb2 as ft
import pandas as pd

from TimeHistory import TimeHistory
from utils import input_output_to_float32, rename_input
from pdb import set_trace


GeneratorData = namedtuple("GeneratorData", "steps gen")


def get_class_names():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    source_data_folder = os.path.join(parent_dir, "Hand_Datasets/EgoGesture_JPG")
    subfolders = os.listdir(source_data_folder)
    classes = [f.split("Single")[-1].lower() for f in subfolders if f.startswith("Single")]
    return classes


def resize_image_keeping_aspect_ratio(image):
    target_size = (640, 480)
    if image.size[0] / image.size[1] > target_size[0] / target_size[1]:
        scale = target_size[0] / image.size[0]
    else:
        scale = target_size[1] / image.size[1]
    new_size = (
        int(round(image.size[0] * scale)),
        int(round(image.size[1] * scale)))
    resized_img = image.resize(new_size)
    paste_rect = (
        int(round((target_size[0] - new_size[0]) / 2)),
        int(round((target_size[1] - new_size[1]) / 2))
    )
    backdrop = Image.new('RGB', target_size, color='black')
    backdrop.paste(resized_img, paste_rect)
    return backdrop


def download_garbage(num_classes=None, limit_per_class=10):
    import subprocess
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    classes_file_path = os.path.join(cur_dir, "class-descriptions-boxable.csv")
    classes_df = pd.read_csv(classes_file_path)
    classes_arr = classes_df.iloc[:, 1].values
    np.random.shuffle(classes_arr)
    if num_classes is not None:
        classes_arr = classes_arr[:num_classes]
    classes_lst = [f"'{cls}'".replace(" ", "_") for cls in classes_arr]
    classes_str = " ".join(classes_lst)
    download_script_folder = os.path.join(cur_dir, "OIDv4_ToolKit")
    garbage_temp_dir = os.path.join(cur_dir, "garbage_temp")
    cd_cmd = f"cd '{download_script_folder}'"
    download_cmd = f" \
    python main.py downloader \
    --Dataset \"{garbage_temp_dir}\" \
    --classes {classes_str} \
    --type_csv train \
    --limit {limit_per_class} \
    "
    download_cmd = download_cmd.strip()
    cmd = f"{cd_cmd} && {download_cmd}"
    os.system(cmd)
    # subprocess.call(cmd, shell=True, executable="/bin/zsh")


def _prepare_garbage_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    source_data_folder = os.path.join(parent_dir, "garbage_raw")
    target_data_folder = os.path.join(parent_dir, "garbage_processed")
    source_folders = [os.path.join(source_data_folder, folder) for folder in os.listdir(source_data_folder)]
    source_files = []

    for folder in source_folders:
        files = os.listdir(folder)
        files = [os.path.join(folder, file) for file in files]
        source_files.extend(files)

    with tqdm(total=len(source_files)) as pbar:
        for idx, file in enumerate(source_files):
            image = Image.open(file)
            image = resize_image_keeping_aspect_ratio(image)
            target_path = os.path.join(target_data_folder, f"{idx}.jpg")
            image.save(target_path, "jpeg")
            pbar.update(1)


def _get_data(classes, batch_size=4, val_size=0.05, test_size=0.05, use_garbage=False):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    data_folder_path = os.path.join(parent_dir, "Hand_Datasets/EgoGesture_JPG")

    data = []
    for idx, cls in enumerate(classes):
        class_path = os.path.join(data_folder_path, f"Single{cls.capitalize()}")
        class_image_paths = [os.path.join(class_path, image_path)
                             for image_path in os.listdir(class_path)]
        class_labels = [idx] * len(class_image_paths)
        class_data = zip(class_image_paths, class_labels)
        data.extend(class_data)

    if use_garbage:
        garbage_folder_path = os.path.join(parent_dir, "garbage_processed")
        garbage_image_paths = [os.path.join(garbage_folder_path, file) for file in os.listdir(garbage_folder_path)]
        class_labels = [len(classes)] * len(garbage_image_paths)
        class_data = list(zip(garbage_image_paths, class_labels))
        data.extend(class_data)

    data = np.array(data, dtype=np.object)
    np.random.shuffle(data)

    val_end = round(len(data) * val_size)
    test_end = val_end + round(len(data) * test_size)

    val_arr = data[:val_end]
    test_arr = data[val_end:test_end]
    train_arr = data[test_end:]

    num_classes = len(classes)
    if use_garbage:
        num_classes += 1

    return [DataGenerator(arr, num_classes=num_classes, batch_size=batch_size) for arr in [train_arr, val_arr, test_arr]]


def create_mobilenet(classes_num):
    model = MobileNetV2(input_shape=(480, 640, 3),
                        weights=None,
                        include_top=True,
                        classes=classes_num)
    return model


def train_mobilenet_1_vs_5():
    train_mobilenet("mobilenet_1_vs_5", warm_start=True, classes=["one", "five"])


def train_mobilenet_1_through_5_with_good_and_bad():
    classes = ["one", "two", "three", "four", "five", "bad", "good"]
    train_mobilenet("mobilenet_1_through_5_with_good_and_bad", warm_start=True, classes=classes)


def train_mobilenet_no_garbage():
    train_mobilenet("mobilenet_no_garbage", warm_start=True, classes=get_class_names())


def train_mobilenet_all():
    train_mobilenet("mobilenet_all", warm_start=True, classes=get_class_names(), use_garbage=True)


def train_mobilenet(model_dir, warm_start, classes, use_garbage=False):
    train_data, val_data, test_data = _get_data(classes, use_garbage=use_garbage)

    from shutil import rmtree
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    model_dir = os.path.join(models_dir, model_dir)
    if not warm_start and os.path.exists(model_dir):
        rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    final_path = os.path.join(model_dir, "final.hdf5")

    num_classes = len(classes)
    if use_garbage:
        num_classes += 1

    model = create_mobilenet(num_classes)

    if warm_start:
        weight_file_to_load = None
        weight_files = [file.rsplit('.', 1)[0] for file in os.listdir(model_dir)]
        if "final" in weight_files:
            weight_file_to_load = "final.hdf5"
        elif len(weight_files) != 0:
            weight_file_to_load = f"{sorted(weight_files)[-1]}.hdf5"
        if weight_file_to_load is not None:
            model.load_weights(os.path.join(model_dir, weight_file_to_load))

    model.compile(optimizer=Adam(lr=0.001),
                  loss=categorical_crossentropy,
                  metrics=["accuracy"])
    model.fit_generator(generator=train_data,
                        epochs=30,
                        verbose=2,
                        callbacks=[
                            ModelCheckpoint(filepath=model_path,
                                            monitor="val_acc",
                                            save_best_only=True,
                                            save_weights_only=True,
                                            verbose=1,
                                            period=1),
                            ReduceLROnPlateau(monitor="val_loss",
                                              factor=0.2,
                                              patience=2,
                                              verbose=1),
                            EarlyStopping(monitor="val_loss",
                                          min_delta=0,
                                          patience=6,
                                          verbose=1,
                                          restore_best_weights=True),
                            TimeHistory()
                        ],
                        validation_data=val_data,
                        )
    model.save_weights(final_path)
    loss, acc = model.evaluate_generator(generator=test_data,
                                         verbose=1)
    print("test_loss:", loss, "test_acc", acc)


def convert_1_vs_5():
    convert_model("mobilenet_1_vs_5", classes=["one", "five"])


def convert_no_garbage():
    convert_model("mobilenet_no_garbage", classes=get_class_names())


def convert_all():
    convert_model("mobilenet_all", classes=get_class_names(), use_garbage=True)


def convert_mobilenet_1_through_5_with_good_and_bad():
    classes = ["one", "two", "three", "four", "five", "bad", "good"]
    convert_model("mobilenet_1_through_5_with_good_and_bad", classes=classes)


def convert_model(model_dir, classes, use_garbage=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    model_dir = os.path.join(models_dir, model_dir)
    if use_garbage:
        classes.append("garbage")
    model = create_mobilenet(len(classes))
    model.load_weights(os.path.join(model_dir, "final.hdf5"))

    coreml_model = coremltools.converters.keras.convert(model=model,
                                                        input_names=["image"],
                                                        output_names=["classProbabilities"],
                                                        class_labels=classes)
    coreml_model_path = os.path.join(model_dir, "GestureNet.mlmodel")
    spec = coreml_model.get_spec()

    metadata = spec.description.metadata
    metadata.author = "Vasilii Dumanov"
    metadata.shortDescription = "MobileNetV2 variant for hand gesture recognition"

    input = spec.description.input[0]
    rename_input(spec, input, "image")
    import coremltools.proto.FeatureTypes_pb2 as ft
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 480
    input.type.imageType.width = 640

    input_output_to_float32(spec)
    coremltools.utils.save_spec(spec, coreml_model_path)


def edit_coreml_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    coreml_model_path = os.path.join(current_dir, "out/GestureNet.mlmodel")
    spec = coremltools.utils.load_spec(coreml_model_path)

    input = spec.description.input[0]
    rename_input(spec, input, "image")
    import coremltools.proto.FeatureTypes_pb2 as ft
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 480
    input.type.imageType.width = 640

    coremltools.utils.save_spec(spec, coreml_model_path)


def main():
    # train_mobilenet_1_through_5_with_good_and_bad()
    convert_mobilenet_1_through_5_with_good_and_bad()


if __name__ == '__main__':
    main()
