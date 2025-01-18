import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10


# 推理函数
def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')  # 加载预训练模型
    if image:
        # 图像推理
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()  # 生成带有标注的图像
        return annotated_image[:, :, ::-1], None  # 返回注释过的图像，将图像的颜色空间从 BGR 转换为 RGB
    else:
        # 视频推理
        video_path = tempfile.mktemp(suffix=".webm")  # 创建一个临时路径，文件后缀是 .webm
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        # 处理视频并保存结果
        cap = cv2.VideoCapture(video_path)  # 使用 OpenCV 读取视频文件，获取帧率、宽度和高度等信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")  # 使用 cv2.VideoWriter 初始化输出视频文件
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)  # 逐帧读取视频并进行推理，得到标注后的帧并写入输出视频文件

        cap.release()
        out.release()

        return None, output_video_path


#  推理函数（示例）
def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image  # 这是一个简化的推理函数，专门用于处理 Gradio 示例，输出标注后的图像


# 创建 Gradio 应用 (app 函数)
def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")  # 这个部分构建了Gradio的前端界面

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        # 控制输入类型的显示
        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video  # 根据选择的输入类型（图像或视频），动态更新可见的组件

        # 执行推理
        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )   # 当 input_type 发生变化时，调用 update_visibility 函数来更新界面

        # 执行推理并返回结果
        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold)

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )  # 根据输入类型（图像或视频），执行相应的推理任务，并返回结果（标注后的图像或视频）

        # 提供示例 (Examples)
        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )  # 提供了一些示例图像，用于展示应用的功能，并设置了延迟加载（cache_examples='lazy'）以提高性能


# 启动应用
gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()  # 使用 gr.Blocks() 初始化并启动 Gradio 应用，显示标题、文献链接等信息，并在用户界面上调用 app() 函数
