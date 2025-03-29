import cv2
from pathlib import Path
from ultralytics import YOLOv10
import argparse


def predict_image(model_path=None, image_path=None, output_dir=None, conf_threshold=0.5, show_result=True):
    """
    使用YOLOv10模型预测单张图片

    参数:
        model_path (str): 模型权重文件路径(.pt)
        image_path (str): 要预测的图片路径
        output_dir (str): 结果保存目录
        conf_threshold (float): 置信度阈值(0-1)
        show_result (bool): 是否显示预测结果
    """
    # 默认参数设置（当没有通过命令行或函数参数传入时使用）
    default_args = {
        'model_path': '../runs/detect/train-v10n+coco128/weights/best.pt',
        'image_path': 'image/000000153011.jpg',
        'output_dir': '../runs/predict',
        'conf_threshold': 0.5,
        'show_result': True
    }

    # 使用传入的参数覆盖默认参数
    if model_path is not None:
        default_args['model_path'] = model_path
    if image_path is not None:
        default_args['image_path'] = image_path
    if output_dir is not None:
        default_args['output_dir'] = output_dir
    if conf_threshold is not None:
        default_args['conf_threshold'] = conf_threshold
    if show_result is not None:
        default_args['show_result'] = show_result

    args = default_args

    # 检查必要参数
    if not args['image_path']:
        raise ValueError("必须提供要预测的图片路径")

    # 创建输出目录
    output_path = Path(args['output_dir'])
    output_path.mkdir(exist_ok=True, parents=True)

    # 加载模型
    model = YOLOv10(args['model_path'])

    # 预测图片
    results = model.predict(
        source=args['image_path'],
        conf=args['conf_threshold'],
        save=True,
        project=args['output_dir'],
        exist_ok=True,
        show=args['show_result']
    )

    # 打印检测结果信息
    for i, result in enumerate(results):
        print(f"\nImage {i + 1} 检测结果:")
        print(f"  路径: {result.path}")
        print(f"  检测到 {len(result.boxes)} 个目标")

        # 打印每个检测目标的详细信息
        for box in result.boxes:
            print(f"  - 类别: {model.names[int(box.cls)]}")
            print(f"    置信度: {box.conf.item():.2f}")
            print(f"    坐标: {[round(x) for x in box.xyxy[0].tolist()]}")

    # 返回结果对象(可用于进一步处理)
    return results


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='YOLOv10单图预测脚本')
    parser.add_argument('--model', type=str, help='模型权重路径')
    parser.add_argument('--image', type=str, help='要预测的图片路径')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--conf', type=float, help='置信度阈值(0-1)')
    parser.add_argument('--show', type=bool, help='是否显示预测结果')
    args = parser.parse_args()

    # 执行预测，使用命令行参数覆盖默认参数
    results = predict_image(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        conf_threshold=args.conf,
        show_result=args.show
    )

    # 获取第一张图片的预测结果
    if results:
        result = results[0]
        output_image = result.save_dir / Path(args.image if args.image else '').name
        print(f"\n预测结果已保存到: {output_image}")