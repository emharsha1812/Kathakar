import gradio as gr
import torch
from transformers import (
    AutoTokenizer as AutoTokenizerSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from transformers import AutoTokenizer
from peft import PeftModel

# 1) Import the IndicTransToolkit
from IndicTransToolkit import IndicProcessor

########################################
# 2) Load the IndicTrans2 translation model
########################################
model_name = "ai4bharat/indictrans2-indic-en-1B"  # Example: English <-> Indian languages
translator_tokenizer = AutoTokenizerSeq2Seq.from_pretrained(model_name, trust_remote_code=True)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
translator_model.eval()

# Instantiate the IndicProcessor for preprocessing & postprocessing
ip = IndicProcessor(inference=True)

# Move translator to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator_model.to(device)

########################################
# 3) Load your story-generation model
########################################
# We'll assume you have your LoRA model in "./final_model"
base_model_name = "sarvamai/sarvam-1"  # or whichever base you used
story_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if story_tokenizer.pad_token is None:
    story_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
base_model.resize_token_embeddings(len(story_tokenizer))
model = PeftModel.from_pretrained(base_model, "./final_model")
model.eval()
model.to(device)

####################################################
# 4) Map from Gradio dropdown label -> IndicTrans2 code
####################################################
# Example mappings for 11 languages plus English
LANG_CODE_MAP = {
    "Hindi (hi)":    "hin_Deva",
    "Bengali (bn)":  "ben_Beng",
    "Gujarati (gu)": "guj_Gujr",
    "Kannada (kn)":  "kan_Knda",
    "Malayalam (ml)": "mal_Mlym",
    "Marathi (mr)":  "mar_Deva",
    "Oriya (or)":    "ory_Orya",
    "Punjabi (pa)":  "pan_Guru",
    "Tamil (ta)":    "tam_Taml",
    "Telugu (te)":   "tel_Telu",
    "English (en)":  "eng_Latn"
}

#####################################
# 5) Translation function: En -> Indic
#####################################
def translate_prompt_to_indic(prompt_text, target_label):
    """Translate an English prompt to the chosen Indian language using IndicTrans2"""
    # If target is English, skip
    if target_label == "English (en)":
        return prompt_text

    # Use the language code from the map
    tgt_lang_code = LANG_CODE_MAP.get(target_label, "eng_Latn")
    # We'll assume the user prompt is in English:
    src_lang_code = "eng_Latn"

    # Preprocess with IndicTrans2 (batch interface expects a list)
    batch = ip.preprocess_batch(
        [prompt_text],
        src_lang=src_lang_code,
        tgt_lang=tgt_lang_code,
    )

    inputs = translator_tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    # Generate translation
    with torch.no_grad():
        generated_tokens = translator_model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1
        )

    # Decode translation
    with translator_tokenizer.as_target_tokenizer():
        output_texts = translator_tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess
    translations = ip.postprocess_batch(output_texts, lang=tgt_lang_code)

    # Return first item if it exists, else fallback
    return translations[0] if translations else prompt_text

#####################################
# 6) Story generation function
#####################################
def generate_story(language_label, prompt_text, theme):
    # Translate the user prompt from English to the chosen Indian language if needed
    translated_prompt = translate_prompt_to_indic(prompt_text, language_label)

    # Build the final input to the model
    model_prompt = f"Write a {theme.lower()} folklore story in {language_label}:\n{translated_prompt}\n"

    input_ids = story_tokenizer(model_prompt, return_tensors="pt").input_ids.to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=story_tokenizer.eos_token_id
        )

    # Decode
    output_text = story_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text

########################################
# 7) Build the Gradio UI
########################################
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Kathakar - Your Regional AI Storyteller (IndicTrans2)")

        with gr.Row():
            language_dropdown = gr.Dropdown(
                label="Select Language",
                choices=list(LANG_CODE_MAP.keys()),
                value="Hindi (hi)",
            )

            prompt_box = gr.Textbox(
                label="Story Prompt (in English)",
                placeholder="E.g., 'A humorous folk tale about a clever crow'"
            )

            thematic_dropdown = gr.Dropdown(
                label="Select Thematic Category",
                choices=["Mythology", "Humor", "Folk Tales", "Moral Stories", "Regional History"],
                value="Mythology"
            )

        generate_button = gr.Button("Generate Story")

        story_output = gr.Textbox(
            label="Generated Story",
            placeholder="Your folklore story will appear here.",
            lines=10
        )

        generate_button.click(
            fn=generate_story,
            inputs=[language_dropdown, prompt_box, thematic_dropdown],
            outputs=story_output,
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(debug=True)
