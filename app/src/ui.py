from __future__ import annotations

from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

from app.src.config import Settings
from app.src.services.internvl_service import InternVLService
from app.src.state.session_store import SessionStore
from app.src.utils.image_ops import DetectedObject, draw_boxes, detections_to_json, image_to_bytes, write_artifact
from app.src.utils.validation import ValidationError, ensure_pil_image, ensure_text


def build_interface(
    settings: Settings,
    service: InternVLService,
    session_store: SessionStore,
) -> gr.Blocks:
    limits = settings.limits

    def _format_history(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return [(user, assistant) for user, assistant in history]

    def handle_image_upload(image: Image.Image | None):
        if image is None:
            return None, "Загрузите изображение, чтобы начать диалог.", gr.update(), gr.update()
        try:
            pil_image = ensure_pil_image(image, limits)
        except ValidationError as exc:
            return None, str(exc), gr.update(), gr.update()
        session_id = session_store.create_session(pil_image)
        return session_id, "Изображение сохранено. Теперь можно задавать вопросы.", [], ""

    def handle_question(image, question, session_id):
        try:
            pil_image = ensure_pil_image(image, limits) if image is not None else None
            result = service.generate_caption(pil_image, question, session_id)
            return (
                result["session_id"],
                _format_history(result["history"]),
                result["answer"],
                gr.update(value=""),
                "",
            )
        except ValidationError as exc:
            return session_id, gr.update(), gr.update(), gr.update(), str(exc)

    def handle_grounding(image, query):
        try:
            pil_image = ensure_pil_image(image, limits)
            text_query = ensure_text(query, "описание", min_chars=3)
            detections = service.find_objects(pil_image, text_query)
            annotated = draw_boxes(pil_image, detections)
            annotated_path = write_artifact(image_to_bytes(annotated, fmt="PNG"), ".png", settings.artifact_dir)
            json_path = write_artifact(
                detections_to_json(detections).encode("utf-8"),
                ".json",
                settings.artifact_dir,
            )
            table = [
                [obj.label, float(f"{obj.confidence:.3f}"), str(list(obj.box))]
                for obj in detections
            ]
            return annotated, table, str(annotated_path), str(json_path), ""
        except ValidationError as exc:
            return None, [], None, None, str(exc)

    with gr.Blocks(css=".error-box {color:#dc2626;font-weight:600;}") as demo:
        gr.Markdown("## InternVL3.5 Demo")
        gr.Markdown(
            "Два сценария: (1) VQA/описание изображения, (2) Поиск объектов по тексту. "
            "Модель и кэш берутся из каталога, примонтированного к контейнеру."
        )

        with gr.Tab("VQA & Captioning"):
            session_state = gr.State(value=None)
            with gr.Row():
                image_input = gr.Image(label="Изображение", type="pil", height=360)
                history = gr.Chatbot(label="История диалога", height=360)
            question_box = gr.Textbox(label="Вопрос", placeholder="Что находится на изображении?", lines=2)
            ask_btn = gr.Button("Задать вопрос")
            answer_box = gr.Textbox(label="Ответ", interactive=False)
            error_box = gr.Markdown("", elem_classes=["error-box"])

            image_input.upload(
                handle_image_upload,
                inputs=[image_input],
                outputs=[session_state, error_box, history, answer_box],
            )
            ask_btn.click(
                handle_question,
                inputs=[image_input, question_box, session_state],
                outputs=[session_state, history, answer_box, question_box, error_box],
            )

        with gr.Tab("Text Object Search"):
            grounding_error = gr.Markdown("", elem_classes=["error-box"])
            with gr.Row():
                grounding_image = gr.Image(label="Изображение", type="pil", height=360)
                detection_preview = gr.Image(label="Результат", height=360)
            grounding_query = gr.Textbox(label="Описание объектов", placeholder="Найди все машины на парковке")
            grounding_button = gr.Button("Найти объекты")
            detections_table = gr.Dataframe(headers=["label", "confidence", "box"], datatype=["str", "number", "str"], interactive=False)
            download_image = gr.File(label="Скачать аннотированное изображение")
            download_json = gr.File(label="Скачать JSON отчет")

            grounding_button.click(
                handle_grounding,
                inputs=[grounding_image, grounding_query],
                outputs=[detection_preview, detections_table, download_image, download_json, grounding_error],
            )

    return demo
