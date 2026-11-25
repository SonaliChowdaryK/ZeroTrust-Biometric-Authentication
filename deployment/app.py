import os
import gradio as gr

DISCLAIMER = (
    "Prototype interface only. The four biometric models and frozen policy engine "
    "are not connected yet. This interface must not be used for real authentication."
)

def status_panel():
    return (
        "M1 Face Anti-Spoofing: NOT CONNECTED\n"
        "M2 Face Recognition: NOT CONNECTED\n"
        "M3 Speaker Recognition: NOT CONNECTED\n"
        "M4 Voice Anti-Spoofing: NOT CONNECTED\n\n"
        "Decision: REQUIRE ADDITIONAL VERIFICATION (safe placeholder)"
    )

with gr.Blocks(title="ZeroTrust Biometric Authentication") as demo:
    gr.Markdown("# ZeroTrust Biometric Authentication")
    gr.Markdown("**Remote development scaffold — biometric inference is not enabled.**")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        gr.File(label="Face image (placeholder upload)", file_types=["image"])
        gr.File(label="Voice recording (placeholder upload)", file_types=["audio"])
    run = gr.Button("Run verification (placeholder)", interactive=True)
    output = gr.Textbox(label="System status / decision", lines=8, value=status_panel())
    run.click(fn=status_panel, inputs=[], outputs=output)
    gr.Markdown(
        "**Privacy:** Do not upload real biometric data to this prototype. "
        "No inference or storage policy is implemented yet."
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
